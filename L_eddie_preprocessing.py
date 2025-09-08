import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import gc # 가비지 컬렉터 (메모리 관리)

def add_time_features_by_spec(
    df: pd.DataFrame,
    time_col: str,
    base_specs: List[Dict[str, Any]],
    *,
    add_time_diff: bool = False,
    enforce_datetime: bool = True,
    leakage_shift: int = 1,
    drop_na: bool = False,
    presorted: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple]]]:
    """
    선언형 스펙으로 시계열 파생피처 생성 (메모리 최종 최적화 버전)

    - 동일한 group_cols, agg 스펙을 묶어서 한 번에 처리하고, merge를 최소화하여
      다중 그룹핑 컬럼 사용 시 발생하는 메모리 문제를 해결합니다.
    """
    # 원본 데이터 보호를 위해 복사본으로 시작
    df_out = df.copy()

    # 시간 컬럼이 datetime 타입이 아니면 변환 (권장)
    if enforce_datetime and not pd.api.types.is_datetime64_any_dtype(df_out[time_col]):
        df_out[time_col] = pd.to_datetime(df_out[time_col], errors="coerce")

    # 데이터가 부족하여 피처 생성이 불가능한 그룹을 기록할 딕셔너리
    insufficient: Dict[str, List[Tuple]] = {}

    # === ⭐️ 핵심 로직 1: 스펙 그룹화 ===
    # 처리 효율을 높이기 위해, 동일한 기준으로 전처리할 수 있는 스펙들을 미리 묶어줍니다.
    spec_groups = defaultdict(list)
    for spec in base_specs:
        grp_cols = tuple(sorted(spec.get("group_cols", [])))
        agg = spec.get("agg")
        source_col = spec["source_col"]
        group_key = (grp_cols, agg, source_col)
        spec_groups[group_key].append(spec)
    
    # --- 피처 생성에 필요한 Helper 함수들 ---
    def _required_points(ops: Dict[str, Any]) -> int:
        cand = [1 + leakage_shift]
        if not ops: return max(cand)
        if ops.get("lag"): cand.append(max(ops["lag"]) + leakage_shift)
        for k in ("rolling_mean", "rolling_std", "rolling_min", "rolling_max"):
            if ops.get(k): cand.append(max(ops[k]) + leakage_shift)
        if ops.get("diff"): cand.append(max(ops["diff"]) + leakage_shift)
        if ops.get("pct_change"): cand.append(max(ops["pct_change"]) + leakage_shift)
        return max(cand)

    def _min_periods(window: int, spec_min_periods: Any) -> int:
        if spec_min_periods in (None, "full"): return window
        try:
            v = int(spec_min_periods)
            return max(1, min(v, window))
        except Exception: return window
    
    # === ⭐️ 핵심 로직 2: 스펙 그룹 단위로 피처 생성 루프 실행 ===
    for (grp_cols_tuple, agg, source_col), specs_in_group in spec_groups.items():
        grp_cols = list(grp_cols_tuple)
        
        # --- STEP 1: '작업대(work_df)' 생성 ---
        base_cols = list(dict.fromkeys(grp_cols + [time_col, source_col]))
        work_df = df_out[base_cols].copy()

        if not presorted:
            sort_keys = grp_cols + [time_col] if grp_cols else [time_col]
            work_df.sort_values(sort_keys, inplace=True)

        agg_source_col = source_col
        group_keys = grp_cols + [time_col]
        
        if agg:
            agg_source_col = f"__agg_{source_col}_{agg}"
            work_df[agg_source_col] = work_df.groupby(group_keys, observed=True)[source_col].transform(agg)
            work_df.drop_duplicates(subset=group_keys, inplace=True)
            if source_col != agg_source_col:
                work_df.drop(columns=[source_col], inplace=True)

        all_new_cols = []
        
        # --- STEP 2: 피처 일괄 계산 ---
        for spec in specs_in_group:
            name: str = spec.get("name") or f"{spec['source_col']}_spec"
            ops: Dict[str, Any] = spec.get("ops", {}) or {}
            spec_min_periods = spec.get("min_periods", "full")
            
            use_dummy = False
            effective_grp_cols = grp_cols
            if not grp_cols:
                use_dummy = True
                dummy = "__all__"
                work_df[dummy] = 1
                effective_grp_cols = [dummy]

            req_n = _required_points(ops)
            sizes = work_df.groupby(effective_grp_cols, dropna=False, observed=True).size()
            bad = sizes[sizes < req_n].index.tolist()
            insufficient[name] = [tuple(x if isinstance(x, tuple) else (x,)) for x in bad]

            g = work_df.groupby(effective_grp_cols, group_keys=False, dropna=False, observed=True)
            
            # === lag 피처 ===
            lag_list = sorted(set(ops.get("lag", []) or []))
            if lag_list:
                s = g[agg_source_col]
                for lag in lag_list:
                    coln = f"{name}_lag{lag}"
                    work_df[coln] = s.shift(lag)
                    all_new_cols.append(coln)

            # === rolling 피처 (mean, std, min, max) ===
            s_shifted = g[agg_source_col].shift(leakage_shift)
            work_df["__tmp_shifted__"] = s_shifted
            for key, func in (("rolling_mean", "mean"), ("rolling_std", "std"),
                              ("rolling_min", "min"), ("rolling_max", "max")):
                wins = ops.get(key, []) or []
                for w in wins:
                    mp = _min_periods(w, spec_min_periods)
                    coln = f"{name}_{key.replace('rolling_', '')}{w}"
                    # GroupBy.rolling은 reset_index가 필요
                    rolled = (
                        work_df.groupby(effective_grp_cols, observed=True)["__tmp_shifted__"]
                            .rolling(w, min_periods=mp)
                            .agg(func)
                            .reset_index(level=list(range(len(effective_grp_cols))), drop=True)
                    )
                    work_df[coln] = rolled
                    all_new_cols.append(coln)
            work_df.drop(columns=["__tmp_shifted__"], inplace=True)

            # === diff 피처 ===
            for k in ops.get("diff", []) or []:
                coln = f"{name}_diff{k}"
                work_df[coln] = g[agg_source_col].transform(lambda s: s.diff(k).shift(leakage_shift))
                all_new_cols.append(coln)

            # === pct_change 피처 ===
            for k in ops.get("pct_change", []) or []:
                coln = f"{name}_pct_change{k}"
                work_df[coln] = g[agg_source_col].transform(lambda s: s.pct_change(k).shift(leakage_shift))
                all_new_cols.append(coln)

            # === ewm (지수이동평균) 피처 ===
            for span in ops.get("ewm_span", []) or []:
                coln = f"{name}_ema{span}"
                work_df[coln] = g[agg_source_col].transform(lambda s: s.shift(leakage_shift).ewm(span=span, adjust=False).mean())
                all_new_cols.append(coln)

            # === lag 간 차이 (diff_from_lags) 피처 ===
            dfl: Union[bool, List[Tuple[int, int]]] = ops.get("diff_from_lags", False)
            if dfl and lag_list:
                pairs = []
                if isinstance(dfl, bool) and len(lag_list) >= 2:
                    pairs = list(zip(lag_list[:-1], lag_list[1:]))
                elif isinstance(dfl, list):
                    pairs = dfl
                
                for a, b in pairs:
                    ca, cb = f"{name}_lag{a}", f"{name}_lag{b}"
                    if ca in work_df.columns and cb in work_df.columns:
                        coln = f"{name}_diff_lag{a}_{b}"
                        work_df[coln] = work_df[ca] - work_df[cb]
                        all_new_cols.append(coln)
            
            # === lag 기반 yoy (yoy_from_lags) 피처 ===
            yoy_cfg = ops.get("yoy_from_lags", False)
            if yoy_cfg and lag_list:
                yoy_pairs, min_months = [], 0
                if isinstance(yoy_cfg, bool):
                    if {1, 13}.issubset(set(lag_list)): yoy_pairs = [(1, 13)]
                elif isinstance(yoy_cfg, list):
                    yoy_pairs = yoy_cfg
                elif isinstance(yoy_cfg, dict):
                    yoy_pairs = yoy_cfg.get("pairs", [])
                    min_months = int(yoy_cfg.get("min_months", 0))

                if yoy_pairs:
                    grp_counts = None
                    if min_months > 0:
                        grp_counts = g[agg_source_col].transform("size")

                    for (a, b) in yoy_pairs:
                        ca, cb = f"{name}_lag{a}", f"{name}_lag{b}"
                        if ca in work_df.columns and cb in work_df.columns:
                            coln = f"{name}_yoy_m1" if (a==1 and b==13) else f"{name}_yoy_lag{a}_{b}"
                            denom = work_df[cb]
                            yoy = (work_df[ca] / denom - 1).where(denom.notna() & (denom != 0))
                            if grp_counts is not None:
                                yoy = yoy.where(grp_counts >= min_months)
                            work_df[coln] = yoy
                            all_new_cols.append(coln)
            
            if use_dummy: # 더미 컬럼 사용 후 정리
                work_df.drop(columns=[dummy], inplace=True)
        
        # --- STEP 3: 단 한 번의 병합(Merge) ---
        if all_new_cols:
            # 중복 생성된 컬럼 제거 (예: 여러 스펙에서 동일한 lag를 요청한 경우)
            all_new_cols = list(dict.fromkeys(all_new_cols))
            merge_cols = group_keys + all_new_cols
            df_out = df_out.merge(
                work_df[merge_cols], 
                on=group_keys, 
                how="left", 
                sort=False
            )

        del work_df, g
        gc.collect()

    if drop_na:
        created = []
        for spec in base_specs:
            prefix = spec.get("name") or f"{spec['source_col']}_spec"
            created += [c for c in df_out.columns if c.startswith(prefix)]
        if created:
            df_out = df_out.dropna(subset=list(dict.fromkeys(created)))

    return df_out, insufficient

# base_specs.append({
#     "source_col": "월별 판매량",
#     "name": "월별_판매량",
#     "group_cols": ["상점ID","상품ID"],
#     "agg": None,                     # ← 평균으로 변경
#     "ops": {
#         "lag": [1, 2, 3],
#         "rolling_mean": [3],           # 3개월 이동평균(현재값 제외; t에서 t-1..t-3)
#         "diff_from_lags": True         # 연속 lag 간 변화량: (lag1-lag2), (lag2-lag3)
#     },
#     "min_periods": "full"
# })


def downcast(df, verbose=True):

    start_mem = df.memory_usage().sum()/ 1024**2
    
    for col in df.columns:
        dtype_name = df[col].dtype.name
        
        if dtype_name == "object" :
            pass
        
        elif dtype_name == "bool" :
            df[col] = df[col].astype('int8')
        
        elif dtype_name.startswith('int') or (df[col].round() == df[col]).all():
            df[col] = pd.to_numeric(df[col], downcast= "integer")
        else:
            df[col] = pd.to_numeric(df[col], downcast="float")
    
    end_mem = df.memory_usage().sum()/1024**2
    if verbose:
        print("{:.1f}% 압축됨".format(100 * (start_mem - end_mem)/ start_mem))  # "{:,1f}" 대신 "{:.1f}"를 사용
    return df

def cat_encoding(df, cat_list, csv_cols=None, csv_paths=None):
    """
    df: 원본 데이터프레임
    cat_list: 라벨 인코딩할 컬럼 리스트
    csv_cols: 매핑 csv로 저장할 컬럼 리스트 (str 또는 list)
    csv_paths: 각 컬럼에 대해 저장할 경로 리스트 (str 또는 list)
    """
    df_copy = df.copy()
    labelencoder = LabelEncoder()
    mappings = {}

    for col in cat_list:
        df_copy[col] = labelencoder.fit_transform(df_copy[col])
        mappings[col] = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))

    # csv_cols/csv_paths: str이면 list로 변환
    if csv_cols is not None and csv_paths is not None:
        if isinstance(csv_cols, str):
            csv_cols = [csv_cols]
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        for c_col, c_path in zip(csv_cols, csv_paths):
            # 매핑 데이터프레임 생성
            mapping_df = pd.DataFrame({
                c_col: list(mappings[c_col].keys()),
                c_col + '_mapping': list(mappings[c_col].values())
            })
            # 폴더 경로가 없으면 생성
            os.makedirs(os.path.dirname(c_path), exist_ok=True)
            # 저장
            mapping_df.to_csv(c_path, index=False, encoding='utf-8-sig')
            print(f"{c_col} 매핑 테이블이 '{c_path}'로 저장되었습니다.")

    for col, mapping in mappings.items():
        print(f"Mapping for {col}: {mapping}")

    return df_copy

def an_target(df, target_col='qty'):
    stats = df[target_col].describe()
    missing_values = df[target_col].isna().sum()
    negative_values = (df[target_col] < 0).sum()
    
    Q1 = stats['25%']
    Q3 = stats['75%']
    IQR = Q3 - Q1
    outliers = ((df[target_col] < (Q1 - 1.5 * IQR)) | (df[target_col] > (Q3 + 1.5 * IQR))).sum()

    sns.boxplot(y=target_col, data=df)
    plt.title(f"Boxplot of {target_col}")
    plt.ylabel(target_col)
    plt.show()

    output_str = f"""
    타깃변수명: {target_col}
    전체개수: {len(df[target_col])}
    최대값: {stats['max']}
    최소값: {stats['min']}
    평균: {stats['mean']}
    미싱값: {missing_values}개
    음수: {negative_values}개
    아웃라이어의 값의 개수: {outliers}
    아웃라이어 비율: {outliers/len(df[target_col])}
    NA 비율: {missing_values/len(df[target_col])}
    """

    print(output_str)

def data_split(
    df, ts_col, target_col,
    train_range=('2020-01-01', '2023-07-31'),
    valid_range=None, test_range=('2023-08-01', '2023-08-31'),
    drop_col=None, id_col=None, dropna=False,
    return_df=False
):
    """
    날짜 기준 시계열 분할 (id별 동기간 분할 지원)
    return_df=True: 분할된 DataFrame (train/valid/test) 반환
    return_df=False: (X_train, y_train, ...) 반환
    """
    # 날짜 변환
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        df[ts_col] = pd.to_datetime(df[ts_col])

    # 드롭 컬럼 처리
    safe_drop = [c for c in (drop_col or []) if c not in [ts_col, target_col]]

    # id별 교집합 필터링 (필요할 때만)
    filtered_df = df
    if id_col:
        # 세 구간 모두에 공통으로 등장하는 id만 남김
        id_set = set(df[id_col].unique())
        for rng in [train_range, valid_range, test_range]:
            if rng is not None:
                s, e = pd.to_datetime(rng[0]), pd.to_datetime(rng[1])
                ids_in_range = set(df[(df[ts_col] >= s) & (df[ts_col] <= e)][id_col].unique())
                id_set &= ids_in_range
        filtered_df = df[df[id_col].isin(id_set)].copy()

    # 날짜 범위 변환
    train_range = (pd.to_datetime(train_range[0]), pd.to_datetime(train_range[1]))
    test_range = (pd.to_datetime(test_range[0]), pd.to_datetime(test_range[1]))
    if valid_range:
        valid_range = (pd.to_datetime(valid_range[0]), pd.to_datetime(valid_range[1]))

    def _mask(rng, ddf):
        s, e = rng
        return (ddf[ts_col] >= s) & (ddf[ts_col] <= e)

    # 분할
    train_df = filtered_df[_mask(train_range, filtered_df)]
    test_df = filtered_df[_mask(test_range, filtered_df)]
    valid_df = filtered_df[_mask(valid_range, filtered_df)] if valid_range else None

    # DataFrame 자체 반환
    if return_df:
        if valid_range:
            return train_df.copy(), valid_df.copy(), test_df.copy()
        else:
            return train_df.copy(), test_df.copy()

    # X/y 분할
    def _split(df_sub):
        X = df_sub.drop([target_col, ts_col] + safe_drop, axis=1)
        y = df_sub[target_col]
        if dropna:
            mask_drop = X.notnull().all(axis=1) & y.notnull()
            X, y = X[mask_drop], y[mask_drop]
        return X, y

    X_train, y_train = _split(train_df)
    X_test, y_test = _split(test_df)
    if valid_range:
        X_valid, y_valid = _split(valid_df)
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test

def remove_outliers_advanced(df, grain_columns=None, target_col='target', method='iqr', threshold=1.5, verbose=True):
    """
    다양한 아웃라이어 감지 방법을 적용하여 이상치를 제거하는 함수.

    Parameters:
    - df (pd.DataFrame): 입력 데이터프레임
    - grain_columns (list, optional): 그룹화할 기준 컬럼 리스트, 기본값 None (전체 데이터 기준)
    - target_col (str): 아웃라이어 판별 대상 컬럼
    - method (str): 사용할 방법 ('iqr', 'zscore', 'modified_zscore' 중 선택)
    - threshold (float): 아웃라이어 감지 기준 (IQR 배수, Z-Score 기준값 등)
    - verbose (bool): 출력 여부

    Returns:
    - df_cleaned (pd.DataFrame): 아웃라이어가 제거된 데이터프레임
    - outlier_info (list): 아웃라이어 감지된 조합 및 정보 리스트
    """
    
    if grain_columns is None or len(grain_columns) == 0:
        grain_columns = []  # 빈 리스트 처리

    outlier_combinations = []
    outlier_indices = set()

    # 그룹별 아웃라이어 탐지
    for values, group in df.groupby(grain_columns) if grain_columns else [(None, df)]:
        data = group[target_col]
        
        if method == 'iqr':
            # ✅ 1. IQR 기반 탐지 (기본값 1.5에서 2.2까지 범위 확장 가능)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

        elif method == 'zscore':
            # ✅ 2. Z-Score 기반 탐지 (정규분포 가정)
            mean = data.mean()
            std_dev = data.std()
            lower_bound = mean - threshold * std_dev
            upper_bound = mean + threshold * std_dev

        elif method == 'modified_zscore':
            # ✅ 3. MAD(중앙절대편차) 기반 수정된 Z-Score 방식
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad  # 0.6745는 정규분포 조정 값
            outliers = data[np.abs(modified_z_scores) > threshold]
            lower_bound, upper_bound = None, None  # Bound 값이 의미 없음 (MAD 방식은 개별 값 기준)

        # 아웃라이어 찾기
        if method in ['iqr', 'zscore']:
            outliers = group[(data < lower_bound) | (data > upper_bound)]
        
        if not outliers.empty:
            outlier_combinations.append({
                "group_values": values,
                "method": method,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "num_outliers": len(outliers)
            })
            outlier_indices.update(outliers.index)

    # 아웃라이어 제거
    df_cleaned = df.drop(index=outlier_indices)

    # 아웃라이어 비율 계산
    outlier_ratio = (len(outlier_indices) / len(df)) * 100 if len(df) > 0 else 0

    # 결과 출력 (verbose 옵션)
    if verbose:
        print(f'아웃라이어 감지된 그룹 개수: {len(outlier_combinations)}')
        print(f'총 아웃라이어 수: {len(outlier_indices)}')
        print(f'아웃라이어 비율: {outlier_ratio:.2f}%')
        print(f'아웃라이어 제거 전 데이터 행 길이: {len(df)}')
        print(f'아웃라이어 제거 후 데이터 행 길이: {len(df_cleaned)}')

    return df_cleaned, outlier_combinations
