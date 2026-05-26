import pandas as pd 
def _map_binary_series(s : pd.Series) -> pd.Series :
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)
    if valset == {'Yes','No'} :
        return s.map({'No':0,'Yes':1}).astype(int)
    if valset == {'Male' , 'Female'}:
        return s.map({'Female':0 , 'Male':1}).astype(int)
    #for any othr 2-category feature
    if len(vals) == 2 :
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0] : 0 , sorted_vals[1] : 1}
        return s.astype(str).map(mapping).astype(int)
    return s 
def build_features(df:pd.DataFrame , target_col:str = "Churn") -> pd.DataFrame :
    df = df.copy()
    obj_cols = [c for c in df.columns if df[c].dtype == 'object' and c != target_col] 
    num_cols = df.select_dtypes(include=['number']).columns.tolist() 
    print(f'categorical columns : {len(obj_cols)} , numerical columns : {len(num_cols)}')
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    print(f'binary columns : {len(binary_cols)} , multi-category columns : {len(multi_cols)}')
    for c in binary_cols :
        df[c] = _map_binary_series(df[c].astype(str))
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols :
        df[bool_cols] = df[bool_cols].astype(int)
    if multi_cols :
        
        df = pd.get_dummies(df , columns=multi_cols , drop_first=True)
    for c in binary_cols :
        if pd.api.types.is_integer_dtype(df[c]) :
            df[c] = df[c].fillna(0).astype(int)
    return df

