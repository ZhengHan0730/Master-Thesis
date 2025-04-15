from flask import request, jsonify, session, send_file, Blueprint
import pandas as pd
import os
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from collections import Counter
from scipy.stats import entropy
import random
import json
from diffprivlib.tools import mean, count_nonzero, quantiles
from diffprivlib.mechanisms import Laplace, Gaussian
import numpy as np
import pandas as pd
from scipy.stats import entropy
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from collections import Counter
import uuid
import io
from datetime import datetime
from scipy.stats import ks_2samp,  pearsonr, spearmanr


bp = Blueprint('main', __name__)

# 存储评估结果（内存示例，可换为数据库）
quality_results = {}

pandas2ri.activate()

sdcmicro = importr('sdcMicro')

def apply_generalization(dataframe, hierarchy_rules):
    generalized_df = dataframe.copy()

    print("Original DataFrame dtypes:")
    print(generalized_df.dtypes)

    for qi, rule in hierarchy_rules.items():
        print(f"Applying generalization for quasi-identifier: {qi}")
        method = rule.get('method', '').lower()

        if method == 'ordering':  # 数值型数据按照区间来分组
            try:
                generalized_df[qi] = pd.to_numeric(generalized_df[qi], errors='coerce')
            except Exception as e:
                print(f"Error converting {qi} to numeric: {e}")
                continue

            layers = rule.get('layers', [])
            default_label = "Other"
            bins = []
            labels = []
            
            for layer in layers:
                try:
                    min_val = float(layer['min'])
                    max_val = float(layer['max'])
                    label = f"{min_val}-{max_val}"
                    bins.append((min_val, max_val))
                    labels.append(label)
                except ValueError as ve:
                    print(f"Error in layer definition for {qi}: {layer}")
                    continue
            
            def assign_label(value):
                if pd.isna(value):
                    return default_label
                for (min_val, max_val), label in zip(bins, labels):
                    if min_val <= value <= max_val:
                        return label
                return default_label

            generalized_df[qi] = generalized_df[qi].apply(assign_label)
            print(f"Generalized {qi} (ordering):")
            print(generalized_df[qi].head())

        elif method == 'dates':  # 日期型数据处理
            try:
                generalized_df[qi] = pd.to_datetime(generalized_df[qi], errors='coerce')
            except Exception as e:
                print(f"Error converting {qi} to datetime: {e}")
                continue

            layers = rule.get('layers', [])
            default_label = "Other"
            date_bins = []
            date_labels = []

            for layer in layers:
                try:
                    min_date = pd.to_datetime(layer['min'])
                    max_date = pd.to_datetime(layer['max'])
                    label = f"{min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
                    date_bins.append((min_date, max_date))
                    date_labels.append(label)
                except Exception as e:
                    print(f"Error in date layer definition for {qi}: {layer}")
                    continue

            def assign_date_label(value):
                if pd.isna(value):
                    return default_label
                for (min_date, max_date), label in zip(date_bins, date_labels):
                    if min_date <= value <= max_date:
                        return label
                return default_label

            generalized_df[qi] = generalized_df[qi].apply(assign_date_label)
            print(f"Generalized {qi} (dates):")
            print(generalized_df[qi].head())

        elif method == 'masking':  # 掩码处理
            masking_string = rule.get('maskingString', 'Masked')
            non_masked_part = masking_string.replace('*', '')
            num_stars = masking_string.count('*')

            def apply_masking(value):
                if value is None or pd.isna(value):
                    return masking_string
                value_str = str(value)
                if len(value_str) <= len(non_masked_part):
                    return '*' * len(value_str)
                return value_str[:len(non_masked_part)] + '*' * num_stars

            generalized_df[qi] = generalized_df[qi].apply(apply_masking)
            print(f"Generalized {qi} (masking):")
            print(generalized_df[qi].head())

        elif method == 'category':  # 分类数据处理
            hierarchy = rule.get('hierarchy', {})
            default_value = rule.get('default', 'Unknown')

            def map_category(value):
                return hierarchy.get(value, default_value)

            generalized_df[qi] = generalized_df[qi].apply(map_category)
            print(f"Generalized {qi} (category):")
            print(generalized_df[qi].head())

        else:
            print(f"Unsupported generalization method for {qi}: {method}")
            continue

    print("Final Generalized DataFrame:")
    print(generalized_df.head())
    return generalized_df

def apply_km_anonymity_r(dataframe, quasi_identifiers, sensitive_column, k_value, m_value,hierarchy_rules,suppression_threshold):
    generalized_df = apply_generalization(dataframe, hierarchy_rules)
    print(generalized_df.head())

    equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')

    generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)

    small_classes_df = generalized_df[generalized_df['class_size'] < k_value]

    num_to_delete = int(len(small_classes_df) * suppression_threshold)
    
    indices_to_drop = small_classes_df.sample(num_to_delete, random_state=42).index

    anonymized_df = generalized_df.drop(indices_to_drop).drop(columns=['class_size'])
    non_compliant_indices = []

    for _, group in anonymized_df.groupby(quasi_identifiers):
        value_counts = group[sensitive_column].value_counts(normalize=True)
        if any(freq > 1 / m_value for freq in value_counts):
            non_compliant_indices.extend(group.index)

    if non_compliant_indices:
        num_to_delete_km = int(len(non_compliant_indices) * suppression_threshold)
        drop_indices_km = pd.Index(non_compliant_indices).to_series().sample(num_to_delete_km, random_state=42).index
        anonymized_df = anonymized_df.drop(drop_indices_km)

    print(anonymized_df.head())
    
    return anonymized_df

def apply_k_anonymity_r(dataframe, quasi_identifiers, k_value, hierarchy_rules, suppression_threshold):
    generalized_df = apply_generalization(dataframe, hierarchy_rules)
    print(generalized_df.head())

    equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')

    generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)
    print(generalized_df,'111',equivalence_class_size) 
    small_classes_df = generalized_df[generalized_df['class_size'] < k_value]

    num_to_delete = int(len(small_classes_df) * suppression_threshold)
    
    indices_to_drop = small_classes_df.sample(num_to_delete, random_state=42).index

    anonymized_df = generalized_df.drop(indices_to_drop).drop(columns=['class_size'])

    print(anonymized_df.head())
    
    return anonymized_df


def apply_t_closeness_r(dataframe, quasi_identifiers, sensitive_attribute, k_value, t_value, hierarchy_rules,suppression_threshold):
    try:
        generalized_df = apply_generalization(dataframe, hierarchy_rules)
        print(generalized_df.head())

        equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')

        generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)

        small_classes_df = generalized_df[generalized_df['class_size'] < k_value]

        num_to_delete = int(len(small_classes_df) * suppression_threshold)
        
        indices_to_drop = small_classes_df.sample(num_to_delete, random_state=42).index

        anonymized_df = generalized_df.drop(indices_to_drop).drop(columns=['class_size'])

        mask = anonymized_df[quasi_identifiers].apply(lambda x: x.str.contains("Masked", na=False)) | anonymized_df[quasi_identifiers].isna()
        anonymized_df = anonymized_df[~mask.any(axis=1)]

        to_remove_indices = set()

        sensitive_attribute = sensitive_attribute[0]
        overall_distribution = calculate_distribution(dataframe, sensitive_attribute)

        equivalence_classes = anonymized_df.groupby(quasi_identifiers)
        for name, group in equivalence_classes:
            print(f"Processing equivalence class: {name}")

            if group[sensitive_attribute].isna().sum() > 0:
                print(f"Equivalence class {name} has missing values for sensitive attribute. Skipping this class.")
                continue 

            if len(group) < 2: 
                continue  
        
            class_distribution = group[sensitive_attribute].value_counts(normalize=True)

            if class_distribution.empty:
                print(f"Equivalence class {name} has an empty distribution. Masking sensitive attribute.")
                anonymized_df.loc[group.index, sensitive_attribute] = 'Masked'
                continue 

            distance = wasserstein_distance(overall_distribution, class_distribution)
            print(f"t-closeness distance for class {name}: {distance}")

            if distance > t_value:
                print(f"Equivalence class {name} exceeds t-closeness threshold (t={t_value}), masking sensitive attribute.")
                to_remove_indices.update(group.index)


        num_to_delete = int(len(to_remove_indices) * suppression_threshold)
        print(111)
        indices_to_delete = random.sample(list(to_remove_indices), num_to_delete)
        print(222)
        anonymized_df = anonymized_df.drop(index=indices_to_delete)

        print(f"Anonymized DataFrame after t-closeness processing (k={k_value}, t={t_value}):")
        print(anonymized_df.head())

        return anonymized_df

    except Exception as e:
        print(f"An error occurred during t-closeness processing: {e}")
        return None

def calculate_distribution(dataframe, sensitive_attribute):
    return dataframe[sensitive_attribute].value_counts(normalize=True)

def calculate_kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    return entropy(p, q)


def apply_l_diversity_r(dataframe, quasi_identifiers, sensitive_attribute, k_value, l_value, hierarchy_rules, suppression_threshold):
    generalized_df = apply_generalization(dataframe, hierarchy_rules)
    print("泛化后的数据框:")
    print(generalized_df.head())

    equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')

    generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)

    small_classes_df = generalized_df[generalized_df['class_size'] < k_value]

    num_to_delete = int(len(small_classes_df) * suppression_threshold)
    
    indices_to_drop = small_classes_df.sample(num_to_delete, random_state=42).index

    anonymized_df = generalized_df.drop(indices_to_drop).drop(columns=['class_size'])

    sensitive_attribute = sensitive_attribute[0]
    non_compliant_indices = []  
    for _, group in anonymized_df.groupby(quasi_identifiers):
        unique_sensitive_values_count = group[sensitive_attribute].nunique()
        print(unique_sensitive_values_count, 'sssss', group)

        if unique_sensitive_values_count < l_value:
            non_compliant_indices.extend(group.index)

    if non_compliant_indices:
        num_to_delete_l = int(len(non_compliant_indices) * suppression_threshold)
        drop_indices_l = pd.Index(non_compliant_indices).to_series().sample(num_to_delete_l, random_state=42).index
        anonymized_df = anonymized_df.drop(drop_indices_l)

    print(anonymized_df.head())
    
    return anonymized_df


def apply_delta_presence(dataframe, quasi_identifiers, sensitive_column, delta_min, delta_max, hierarchy_rules, suppression_threshold):
    try:
        # 第一步：应用泛化规则
        generalized_df = apply_generalization(dataframe, hierarchy_rules)
        if generalized_df is None or generalized_df.empty:
            raise ValueError("泛化后的数据框为空，请检查泛化规则和输入数据。")
        
        print("泛化后的数据框:")
        print(generalized_df.head())

        # 第二步：计算等价类大小
        if not all(qi in generalized_df.columns for qi in quasi_identifiers):
            raise ValueError(f"某些准标识符 {quasi_identifiers} 不存在于数据框中。")
        
        equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')
        generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)
        print("等价类大小计算完成。")

        # 第三步：检查等价类是否违反 δ-Presence 约束
        non_compliant_indices = []

        for _, group in generalized_df.groupby(quasi_identifiers):
            if sensitive_column not in group.columns:
                raise ValueError(f"敏感列 {sensitive_column} 不存在于数据框中。")
            
            sensitive_value_counts = group[sensitive_column].value_counts(normalize=True)

            if sensitive_value_counts.empty:
                print(f"等价类 {group[quasi_identifiers]} 没有敏感属性值。跳过检查。")
                continue

            # 检查等价类中敏感属性分布是否违反 δ-Presence
            if sensitive_value_counts.max() > delta_max or sensitive_value_counts.min() < delta_min:
                print(f"等价类 {group[quasi_identifiers]} 违反了 δ-Presence 约束。")
                non_compliant_indices.extend(group.index)

        # 第四步：对不满足条件的记录进行抑制
        if non_compliant_indices:
            num_to_delete = max(1, int(len(non_compliant_indices) * suppression_threshold))  # 确保至少删除一条记录
            indices_to_drop = random.sample(non_compliant_indices, num_to_delete)
            anonymized_df = generalized_df.drop(indices_to_drop)
        else:
            anonymized_df = generalized_df

        if anonymized_df.empty:
            raise ValueError("所有数据都被抑制，请降低 suppression_threshold 或调整 δ-Presence 参数。")

        print(f"δ-Presence 处理后的匿名化数据框 (δ_min={delta_min}, δ_max={delta_max}):")
        print(anonymized_df.head())

        return anonymized_df.drop(columns=['class_size'])

    except Exception as e:
        print(f"δ-Presence 处理时发生错误: {e}")
        return None
    

def apply_beta_likeness_r(dataframe, quasi_identifiers, sensitive_attribute, beta_value, hierarchy_rules, suppression_threshold):
    try:
        # Step 1: Apply generalization
        generalized_df = apply_generalization(dataframe, hierarchy_rules)
        print("Generalized DataFrame:")
        print(generalized_df.head())

        # Step 2: Calculate equivalence class sizes
        equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')
        generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)

        # Step 3: Remove small equivalence classes (if necessary)
        small_classes_df = generalized_df[generalized_df['class_size'] < 2]
        num_to_delete = int(len(small_classes_df) * suppression_threshold)
        indices_to_drop = small_classes_df.sample(num_to_delete, random_state=42).index
        anonymized_df = generalized_df.drop(indices_to_drop).drop(columns=['class_size'])

        # Step 4: Calculate global sensitive attribute distribution
        sensitive_attribute = sensitive_attribute[0]  # Ensure it's a single column name
        global_distribution = calculate_distribution(dataframe, sensitive_attribute)
        print("Global Sensitive Attribute Distribution:")
        print(global_distribution)

        # Step 5: Enforce β-likeness
        to_remove_indices = set()
        equivalence_classes = anonymized_df.groupby(quasi_identifiers)

        for name, group in equivalence_classes:
            # Calculate equivalence class distribution
            class_distribution = group[sensitive_attribute].value_counts(normalize=True)
            print(f"Equivalence Class {name} Distribution:")
            print(class_distribution)

            # Compute Wasserstein distance between global and class distributions
            distance = wasserstein_distance(
                global_distribution.values, class_distribution.reindex(global_distribution.index, fill_value=0).values
            )
            print(f"β-likeness distance for class {name}: {distance}")

            # If distance exceeds β-value, mark the class for suppression
            if distance > beta_value:
                print(f"Equivalence class {name} exceeds β-likeness threshold (β={beta_value}). Masking sensitive attribute.")
                anonymized_df.loc[group.index, sensitive_attribute] = "Masked"

        # Step 6: Suppress or remove non-compliant records
        mask_count = anonymized_df[sensitive_attribute].value_counts().get("Masked", 0)
        print(f"Total masked records: {mask_count}")

        if mask_count > 0:
            num_to_delete = int(mask_count * suppression_threshold)
            indices_to_delete = anonymized_df[anonymized_df[sensitive_attribute] == "Masked"].sample(
                num_to_delete, random_state=42
            ).index
            anonymized_df = anonymized_df.drop(index=indices_to_delete)

        print(f"Anonymized DataFrame after β-likeness processing (β={beta_value}):")
        print(anonymized_df.head())

        return anonymized_df

    except Exception as e:
        print(f"An error occurred during β-likeness processing: {e}")
        return None



def apply_delta_disclosure(dataframe, quasi_identifiers, sensitive_attribute, delta_value, hierarchy_rules, suppression_threshold):
    try:
        # Step 1: Apply generalization
        generalized_df = apply_generalization(dataframe, hierarchy_rules)
        print("Generalized DataFrame:")
        print(generalized_df.head())

        # Step 2: Calculate equivalence class sizes
        equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')
        generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)

        # Step 3: Remove very small equivalence classes first
        small_classes_df = generalized_df[generalized_df['class_size'] < 2]
        num_to_delete = int(len(small_classes_df) * suppression_threshold)
        indices_to_drop = small_classes_df.sample(num_to_delete, random_state=42).index
        anonymized_df = generalized_df.drop(indices_to_drop).drop(columns=['class_size'])

        # Step 4: Calculate disclosure risk for each equivalence class
        non_compliant_indices = []
        sensitive_attribute = sensitive_attribute[0]  # Get the first sensitive attribute

        for name, group in anonymized_df.groupby(quasi_identifiers):
            # Calculate conditional probabilities for each sensitive value
            sensitive_value_counts = group[sensitive_attribute].value_counts()
            total_records = len(group)
            
            # Check if any sensitive value can be inferred with probability > delta
            max_probability = sensitive_value_counts.max() / total_records
            
            if max_probability > delta_value:
                print(f"Equivalence class {name} exceeds δ-disclosure threshold (δ={delta_value})")
                print(f"Max probability: {max_probability}")
                non_compliant_indices.extend(group.index)

        # Step 5: Apply suppression to non-compliant records
        if non_compliant_indices:
            num_to_delete = int(len(non_compliant_indices) * suppression_threshold)
            indices_to_delete = random.sample(non_compliant_indices, num_to_delete)
            anonymized_df = anonymized_df.drop(index=indices_to_delete)

        print(f"Anonymized DataFrame after δ-disclosure processing (δ={delta_value}):")
        print(anonymized_df.head())

        return anonymized_df

    except Exception as e:
        print(f"An error occurred during δ-disclosure processing: {e}")
        return None
    

def apply_p_sensitivity(dataframe, quasi_identifiers, sensitive_attribute, p_value, hierarchy_rules, suppression_threshold):
    try:
        # Step 1: Apply generalization
        generalized_df = apply_generalization(dataframe, hierarchy_rules)
        print("Generalized DataFrame:")
        print(generalized_df.head())

        # Step 2: Calculate equivalence class sizes
        equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')
        generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)

        # Step 3: Process small equivalence classes
        small_classes_df = generalized_df[generalized_df['class_size'] < 2]
        num_to_delete = int(len(small_classes_df) * suppression_threshold)
        if num_to_delete > 0:
            indices_to_drop = small_classes_df.sample(num_to_delete, random_state=42).index
            anonymized_df = generalized_df.drop(indices_to_drop).drop(columns=['class_size'])
        else:
            anonymized_df = generalized_df.drop(columns=['class_size'])

        # Step 4: Check p-sensitivity for each equivalence class
        non_compliant_indices = []
        sensitive_attribute = sensitive_attribute[0]  # Get the first sensitive attribute

        for name, group in anonymized_df.groupby(quasi_identifiers):
            unique_sensitive_values = group[sensitive_attribute].nunique()
            if unique_sensitive_values < p_value:
                print(f"Equivalence class {name} has {unique_sensitive_values} unique sensitive values, less than required p={p_value}")
                non_compliant_indices.extend(group.index)

        # Step 5: Apply suppression to non-compliant records
        if non_compliant_indices:
            num_to_delete = int(len(non_compliant_indices) * suppression_threshold)
            if num_to_delete > 0:
                indices_to_delete = random.sample(non_compliant_indices, num_to_delete)
                anonymized_df = anonymized_df.drop(index=indices_to_delete)

        print(f"Anonymized DataFrame after p-sensitivity processing (p={p_value}):")
        print(anonymized_df.head())
        print(f"Final dataset size: {len(anonymized_df)} rows")

        return anonymized_df

    except Exception as e:
        print(f"An error occurred during p-sensitivity processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def apply_ck_safety(dataframe, quasi_identifiers, sensitive_attribute, c_value, k_value, hierarchy_rules, suppression_threshold):
    """
    Apply (c,k)-Safety to the dataset.
    
    Parameters:
    - dataframe: Input DataFrame
    - quasi_identifiers: List of quasi-identifier columns
    - sensitive_attribute: Sensitive attribute column
    - c_value: Maximum confidence threshold (in percentage, e.g., 60 for 60%)
    - k_value: Minimum size of equivalence class
    - hierarchy_rules: Dictionary containing generalization hierarchies
    - suppression_threshold: Threshold for record suppression
    
    Returns:
    - Anonymized DataFrame satisfying (c,k)-Safety
    """
    try:
        # Step 1: Apply generalization
        generalized_df = apply_generalization(dataframe, hierarchy_rules)
        print("Generalized DataFrame:")
        print(generalized_df.head())

        # Step 2: Calculate equivalence class sizes
        equivalence_class_size = generalized_df.groupby(quasi_identifiers).size().reset_index(name='class_size')
        generalized_df = generalized_df.merge(equivalence_class_size, on=quasi_identifiers)

        # Step 3: Remove equivalence classes smaller than k
        small_classes_df = generalized_df[generalized_df['class_size'] < k_value]
        num_to_delete = int(len(small_classes_df) * suppression_threshold)
        indices_to_drop = small_classes_df.sample(num_to_delete, random_state=42).index
        anonymized_df = generalized_df.drop(indices_to_drop).drop(columns=['class_size'])

        # Step 4: Check confidence threshold for each equivalence class
        non_compliant_indices = []
        sensitive_attribute = sensitive_attribute[0]  # Get the first sensitive attribute

        for name, group in anonymized_df.groupby(quasi_identifiers):
            # Calculate confidence for each sensitive value in the group
            value_counts = group[sensitive_attribute].value_counts()
            total_records = len(group)
            
            # Check if any sensitive value exceeds the confidence threshold
            max_confidence = (value_counts.max() / total_records) * 100
            
            if max_confidence > c_value:
                print(f"Equivalence class {name} exceeds confidence threshold (c={c_value}%)")
                print(f"Max confidence: {max_confidence}%")
                non_compliant_indices.extend(group.index)

        # Step 5: Apply suppression to records that violate c-confidence
        if non_compliant_indices:
            num_to_delete = int(len(non_compliant_indices) * suppression_threshold)
            indices_to_delete = random.sample(non_compliant_indices, num_to_delete)
            anonymized_df = anonymized_df.drop(index=indices_to_delete)

        print(f"Anonymized DataFrame after (c,k)-Safety processing (c={c_value}%, k={k_value}):")
        print(anonymized_df.head())
        print(f"Final dataset size: {len(anonymized_df)} rows")

        return anonymized_df

    except Exception as e:
        print(f"An error occurred during (c,k)-Safety processing: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def apply_differential_privacy(dataframe, epsilon, delta, quasi_identifiers, sensitive_columns, hierarchy_rules, budget=1.0, suppression_threshold=0.3):
    """
    Apply differential privacy to the dataset using diffprivlib library.
    """
    try:
        # Create a copy of the original dataframe
        anonymized_df = dataframe.copy()
        
        # First apply generalization to reduce dimensionality
        generalized_df = apply_generalization(dataframe, hierarchy_rules)
        
        # Calculate the budget distribution among columns
        qi_budget = budget * 0.5 / len(quasi_identifiers) if quasi_identifiers else 0
        sensitive_budget = budget * 0.5 / len(sensitive_columns) if sensitive_columns else 0
        
        for column in generalized_df.columns:
            try:
                col_data = pd.to_numeric(generalized_df[column], errors='coerce')
                is_numeric = not col_data.isna().all()
            except:
                is_numeric = False
            
            if is_numeric:
                col_data = pd.to_numeric(generalized_df[column], errors='coerce')
                
                if col_data.isna().sum() / len(col_data) > 0.5:
                    anonymized_df[column] = generalized_df[column]
                    continue
                
                clean_data = col_data.dropna()
                data_min = clean_data.min()
                data_max = clean_data.max()
                sensitivity = (data_max - data_min) * 0.01
                
                if column in quasi_identifiers:
                    col_epsilon = epsilon * qi_budget
                    laplace_mech = Laplace(epsilon=col_epsilon, sensitivity=sensitivity)
                    noisy_data = col_data.copy()
                    for idx in clean_data.index:
                        noisy_data.at[idx] = laplace_mech.randomise(clean_data.at[idx])
                    anonymized_df[column] = noisy_data
                
                elif column in sensitive_columns:
                    col_epsilon = epsilon * sensitive_budget
                    gaussian_mech = Gaussian(epsilon=col_epsilon, delta=delta, sensitivity=sensitivity)
                    noisy_data = col_data.copy()
                    for idx in clean_data.index:
                        noisy_data.at[idx] = gaussian_mech.randomise(clean_data.at[idx])
                    anonymized_df[column] = noisy_data
                
                else:
                    anonymized_df[column] = generalized_df[column]
            
            else:
                if column in quasi_identifiers or column in sensitive_columns:
                    unique_values = generalized_df[column].dropna().unique()
                    if len(unique_values) > 1:
                        replace_prob = np.minimum(0.5, 1.0 / epsilon)
                        noisy_data = generalized_df[column].copy()
                        for idx in range(len(noisy_data)):
                            if pd.notna(noisy_data.iloc[idx]) and np.random.random() < replace_prob:
                                replacement_idx = np.random.choice(len(unique_values))
                                noisy_data.iloc[idx] = unique_values[replacement_idx]
                        anonymized_df[column] = noisy_data
                    else:
                        anonymized_df[column] = generalized_df[column]
                else:
                    anonymized_df[column] = generalized_df[column]
        
        if suppression_threshold > 0:
            suppress_mask = np.random.random(size=len(anonymized_df)) < suppression_threshold
            for column in anonymized_df.columns:
                if column in quasi_identifiers:
                    anonymized_df.loc[suppress_mask, column] = "Masked"
                elif column in sensitive_columns:
                    anonymized_df.loc[suppress_mask, column] = "Masked"
        
        print(f"Differentially Private DataFrame (epsilon={epsilon}, delta={delta}):")
        print(anonymized_df.head())
        
        return anonymized_df
    
    except Exception as e:
        import traceback
        print("Error in differential privacy application:")
        traceback.print_exc()
        print(f"Error details: {str(e)}")
        return None

    
def dp_variance(data, epsilon=1.0):
    """
    计算数据的差分隐私方差
    :param data: 需要计算方差的数据列表
    :param epsilon: 隐私预算
    :return: 计算出的 DP 方差
    """
    dp_mean_value = mean(data, epsilon=epsilon / 2)  # 计算 DP 均值
    dp_square_mean = mean(np.square(data), epsilon=epsilon / 2)  # 计算平方的均值
    return dp_square_mean - dp_mean_value**2  # 方差公式：E(X²) - (E(X))²

def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError("Unsupported file format. Please use a .csv or .tsv file.")

# data quality evaluation method
def allowed_file(filename):
    return filename.endswith('.csv') or filename.endswith('.tsv')

def calculate_mean_diff(original_df, anonymized_df, columns_to_compare):
    result = []
    for col in columns_to_compare:
        try:
            orig_col = pd.to_numeric(original_df[col], errors='coerce')
            anon_col = pd.to_numeric(anonymized_df[col], errors='coerce')
            orig_mean = orig_col.mean()
            anon_mean = anon_col.mean()
            mean_diff = abs(orig_mean - anon_mean)
            result.append({
                'column': col,
                'metric': 'mean',
                'original': round(orig_mean, 4),
                'anonymized': round(anon_mean, 4),
                'difference': round(mean_diff, 4)
            })
        except Exception as e:
            result.append({'column': col, 'metric': 'mean', 'error': str(e)})
    return result

def calculate_median_diff(original_df, anonymized_df, columns_to_compare):
    result = []
    for col in columns_to_compare:
        try:
            orig_col = pd.to_numeric(original_df[col], errors='coerce')
            anon_col = pd.to_numeric(anonymized_df[col], errors='coerce')
            orig_median = orig_col.median()
            anon_median = anon_col.median()
            median_diff = abs(orig_median - anon_median)
            result.append({
                'column': col,
                'metric': 'median',
                'original': round(orig_median, 4),
                'anonymized': round(anon_median, 4),
                'difference': round(median_diff, 4)
            })
        except Exception as e:
            result.append({'column': col, 'metric': 'median', 'error': str(e)})
    return result

def calculate_variance_diff(original_df, anonymized_df, columns_to_compare):
    result = []
    for col in columns_to_compare:
        try:
            orig_col = pd.to_numeric(original_df[col], errors='coerce')
            anon_col = pd.to_numeric(anonymized_df[col], errors='coerce')
            orig_var = orig_col.var()
            anon_var = anon_col.var()
            var_diff = abs(orig_var - anon_var)
            result.append({
                'column': col,
                'metric': 'variance',
                'original': round(orig_var, 4),
                'anonymized': round(anon_var, 4),
                'difference': round(var_diff, 4)
            })
        except Exception as e:
            result.append({'column': col, 'metric': 'variance', 'error': str(e)})
    return result

# wasserstein
def calculate_wasserstein_distance(original_df, anonymized_df, columns_to_compare):
    """
    计算原始数据与匿名化数据在指定列上的 Wasserstein 距离（用于数值型字段）

    参数：
    - original_df: 原始 DataFrame
    - anonymized_df: 匿名化后的 DataFrame
    - columns_to_compare: 要比较的列名列表

    返回：
    - 距离结果列表，每列对应一个 dict（结构与 mean/variance 相同）
    """
    results = []

    for col in columns_to_compare:
        try:
            orig_col = pd.to_numeric(original_df[col], errors='coerce').dropna()
            anon_col = pd.to_numeric(anonymized_df[col], errors='coerce').dropna()

            if len(orig_col) < 2 or len(anon_col) < 2:
                results.append({
                    'column': col,
                    'metric': 'wasserstein',
                    'original': None,
                    'anonymized': None,
                    'difference': None,
                    'error': '数据样本量不足'
                })
                continue

            distance = wasserstein_distance(orig_col, anon_col)
            results.append({
                'column': col,
                'metric': 'wasserstein',
                'original': round(orig_col.mean(), 4),
                'anonymized': round(anon_col.mean(), 4),
                'difference': round(distance, 4)
            })
        except Exception as e:
            results.append({
                'column': col,
                'metric': 'wasserstein',
                'original': None,
                'anonymized': None,
                'difference': None,
                'error': str(e)
            })

    return results

# KS-similarity
def calculate_ks_similarity(original_df, anonymized_df, columns_to_compare):
    """
    计算原始数据与匿名化数据在指定列上的 Kolmogorov-Smirnov (KS) 相似度
    原理：1 - KS statistic，即越接近 1 相似度越高

    返回结构与其他指标一致
    """
    results = []

    for col in columns_to_compare:
        try:
            orig_col = pd.to_numeric(original_df[col], errors='coerce').dropna()
            anon_col = pd.to_numeric(anonymized_df[col], errors='coerce').dropna()

            if len(orig_col) < 2 or len(anon_col) < 2:
                results.append({
                    'column': col,
                    'metric': 'ks_similarity',
                    'original': None,
                    'anonymized': None,
                    'difference': None,
                    'error': '数据样本量不足'
                })
                continue

            stat, _ = ks_2samp(orig_col, anon_col)
            similarity = 1 - stat
            results.append({
                'column': col,
                'metric': 'ks_similarity',
                'original': round(orig_col.mean(), 4),
                'anonymized': round(anon_col.mean(), 4),
                'difference': round(similarity, 4)
            })
        except Exception as e:
            results.append({
                'column': col,
                'metric': 'ks_similarity',
                'original': None,
                'anonymized': None,
                'difference': None,
                'error': str(e)
            })

    return results
# pearson
def calculate_pearson(original_df, anonymized_df, columns_to_compare):
    results = []

    for col in columns_to_compare:
        try:
            orig_col = pd.to_numeric(original_df[col], errors='coerce')
            anon_col = pd.to_numeric(anonymized_df[col], errors='coerce')

            # 筛掉缺失值
            valid_index = orig_col.notna() & anon_col.notna()
            orig_col = orig_col[valid_index]
            anon_col = anon_col[valid_index]

            if len(orig_col) < 2:
                results.append({
                    'column': col,
                    'metric': 'pearson',
                    'original': None,
                    'anonymized': None,
                    'difference': None,
                    'error': '数据样本量不足'
                })
                continue

            pearson_corr, _ = pearsonr(orig_col, anon_col)
            results.append({
                'column': col,
                'metric': 'pearson',
                'original': round(orig_col.mean(), 4),
                'anonymized': round(anon_col.mean(), 4),
                'difference': round(pearson_corr, 4)
            })

        except Exception as e:
            results.append({
                'column': col,
                'metric': 'pearson',
                'original': None,
                'anonymized': None,
                'difference': None,
                'error': str(e)
            })

    return results

# spearman
def calculate_spearman(original_df, anonymized_df, columns_to_compare):
    results = []

    for col in columns_to_compare:
        try:
            orig_col = pd.to_numeric(original_df[col], errors='coerce')
            anon_col = pd.to_numeric(anonymized_df[col], errors='coerce')

            valid_index = orig_col.notna() & anon_col.notna()
            orig_col = orig_col[valid_index]
            anon_col = anon_col[valid_index]

            if len(orig_col) < 2:
                results.append({
                    'column': col,
                    'metric': 'spearman',
                    'original': None,
                    'anonymized': None,
                    'difference': None,
                    'error': '数据样本量不足'
                })
                continue

            spearman_corr, _ = spearmanr(orig_col, anon_col)
            results.append({
                'column': col,
                'metric': 'spearman',
                'original': round(orig_col.mean(), 4),
                'anonymized': round(anon_col.mean(), 4),
                'difference': round(spearman_corr, 4)
            })
        except Exception as e:
            results.append({
                'column': col,
                'metric': 'spearman',
                'original': None,
                'anonymized': None,
                'difference': None,
                'error': str(e)
            })

    return results



# 区间值预处理
def parse_range(value):
    """处理各种不同格式的区间值，返回中点值"""
    # 如果是None或不是字符串，尝试直接转为浮点数
    if not isinstance(value, str):
        try:
            return float(value)
        except:
            return np.nan
    
    # 处理 "x-y" 格式
    if '-' in value:
        try:
            start, end = map(float, value.split('-'))
            return (start + end) / 2
        except:
            pass
    
    # 处理 "[x,y]" 或 "(x,y)" 格式
    if (',' in value) and (value.startswith('[') or value.startswith('(')) and (value.endswith(']') or value.endswith(')')):
        try:
            # 移除括号
            clean_value = value.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            start, end = map(float, clean_value.split(','))
            return (start + end) / 2
        except:
            pass
    
    # 尝试直接转换为浮点数
    try:
        return float(value)
    except:
        return np.nan

# 新增：用于将原始日期或匿名化日期区间转为距今天数
def convert_date_to_days(column, is_range=False):
    """
    将日期列转换为距今的天数：
    - 若 is_range=True，则处理为区间（如 '2009-01-01 - 2024-07-01'），取中点计算距今天数
    - 若 is_range=False，则为普通单个日期字段
    """
    today = datetime.today()

    if is_range:
        def parse_date_range(date_str):
            try:
                if pd.isna(date_str):
                    return None
                start_str, end_str = date_str.split(' - ')
                start = pd.to_datetime(start_str.strip(), errors='coerce')
                end = pd.to_datetime(end_str.strip(), errors='coerce')
                if pd.isna(start) or pd.isna(end):
                    return None
                midpoint = start + (end - start) / 2
                return (today - midpoint).days
            except Exception as e:
                print(f"日期区间解析失败: {date_str}，错误: {e}")
                return None

        return column.apply(parse_date_range)

    else:
        column = pd.to_datetime(column, errors='coerce')
        return (today - column).dt.days
   

# 对列进行预处理
def preprocess_column(column):
    """
    自动识别并处理列数据：
    - 日期或日期区间：调用 convert_date_to_days
    - 数值区间或普通数字：调用 parse_range
    """
    sample_value = column.dropna().astype(str).iloc[0] if not column.dropna().empty else ''

    # 简单规则判断是否为日期或日期区间（包含年信息 + 分隔符）
    if any(ch in sample_value for ch in ['-', '/']) and any(kw in sample_value for kw in ['20', '19']):
        is_range = ' - ' in sample_value
        return convert_date_to_days(column, is_range=is_range)

    # 其余默认按数值区间处理
    return column.apply(parse_range)



@bp.route('/evaluation', methods=['POST']) 
def data_quality_evaluation():     
    if 'original_file' not in request.files or 'anonymized_file' not in request.files:         
        return jsonify({'error': '必须同时上传原始文件和匿名化文件'}), 400      
    
    original_file = request.files['original_file']     
    anonymized_file = request.files['anonymized_file']     
    columns_to_compare = request.form.get('columns')     
    metrics = request.form.get('metrics', 'mean,median,variance').split(',')      
    
    if not columns_to_compare:         
        return jsonify({'error': '必须指定用于比较的列'}), 400      
    
    columns_to_compare = columns_to_compare.split(',')      
    
    # 文件扩展名校验     
    if not allowed_file(original_file.filename) or not allowed_file(anonymized_file.filename):         
        return jsonify({'error': '只支持 .csv 或 .tsv 文件格式'}), 400      
    
    try:         
        original_df = pd.read_csv(original_file, on_bad_lines='skip')         
        anonymized_df = pd.read_csv(anonymized_file, on_bad_lines='skip')     
    except Exception as e:         
        return jsonify({'error': f'文件读取失败，请检查格式：{str(e)}'}), 400      
    
    # 检查列是否存在     
    missing_cols = [col for col in columns_to_compare if col not in original_df.columns or col not in anonymized_df.columns]     
    if missing_cols:         
        return jsonify({'error': f'以下列在文件中不存在: {missing_cols}'}), 400
    
    # 如果选择了数值统计方法，先验证数值类型，对非数值类型尝试区间处理
    if any(metric in metrics for metric in ['mean', 'median', 'variance', 'wasserstein', 'ks_similarity', 'pearson', 'spearman']):
        # 收集非数值类型的列
        non_numeric_cols = [
            col for col in columns_to_compare
            if not pd.api.types.is_numeric_dtype(original_df[col]) or not pd.api.types.is_numeric_dtype(anonymized_df[col])
        ]
        
        # 对非数值类型的列尝试预处理
        for col in non_numeric_cols:
            try:
                # 预处理并生成新列
                processed_col_name = col + '_processed'
                original_df[processed_col_name] = preprocess_column(original_df[col])
                anonymized_df[processed_col_name] = preprocess_column(anonymized_df[col])

                # 强制转换为 float 确保后续处理无误
                original_df[processed_col_name] = pd.to_numeric(original_df[processed_col_name], errors='coerce').astype(float)
                anonymized_df[processed_col_name] = pd.to_numeric(anonymized_df[processed_col_name], errors='coerce').astype(float)

                # 更新列名用于后续比较
                columns_to_compare = [processed_col_name if c == col else c for c in columns_to_compare]

            except Exception as e:
                return jsonify({'error': f'数据预处理失败: {col}, 错误: {str(e)}'}), 400

        
        # 再次验证是否有非数值类型的列
        numeric_check_failed = [
            col for col in columns_to_compare
            if not pd.api.types.is_numeric_dtype(original_df[col]) or not pd.api.types.is_numeric_dtype(anonymized_df[col])
        ]
        
        if numeric_check_failed:
            return jsonify({'error': f'以下列不是数值类型，无法计算统计指标: {numeric_check_failed}'}), 400      
    
    results = []     
    if 'mean' in metrics:         
        results.extend(calculate_mean_diff(original_df, anonymized_df, columns_to_compare))     
    if 'median' in metrics:         
        results.extend(calculate_median_diff(original_df, anonymized_df, columns_to_compare))     
    if 'variance' in metrics:         
        results.extend(calculate_variance_diff(original_df, anonymized_df, columns_to_compare))  
    if 'wasserstein' in metrics:
        results.extend(calculate_wasserstein_distance(original_df, anonymized_df, columns_to_compare))
    if 'ks_similarity' in metrics:
        results.extend(calculate_ks_similarity(original_df, anonymized_df, columns_to_compare))
    if 'pearson' in metrics:
        results.extend(calculate_pearson(original_df, anonymized_df, columns_to_compare))
    if 'spearman' in metrics:
        results.extend(calculate_spearman(original_df, anonymized_df, columns_to_compare))
    
    result_id = str(uuid.uuid4())     
    quality_results[result_id] = results      
    
    return jsonify({'result_id': result_id, 'summary': results})

# 获取评估后结果
@bp.route('/quality/result/<result_id>', methods=['GET'])
def get_quality_result(result_id):
    if result_id in quality_results:
        return jsonify(quality_results[result_id])
    else:
        return jsonify({'error': 'Result ID not found'}), 404

# 评估结果下载
@bp.route('/quality/result/<result_id>/download', methods=['GET'])
def download_quality_result(result_id):
    if result_id not in quality_results:
        return jsonify({'error': 'Result ID not found'}), 404

    result_df = pd.DataFrame(quality_results[result_id])
    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'quality_result_{result_id}.csv'
    )





@bp.route('/anonymize', methods=['POST'])
def anonymize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    privacy_model = request.form.get('privacy_model', 'k-anonymity')
    k_value = request.form.get('k', None)
    if k_value:
        k_value = int(k_value)
    
    l_value = request.form.get('l', None)
    if l_value:
        l_value = int(l_value)
    
    t_value = request.form.get('t', None)
    if t_value:
        t_value = float(t_value)
    
    m_value = request.form.get('m', None)
    if m_value:
        m_value = float(m_value)

    delta_min = request.form.get('delta_min', None)
    if delta_min:
        delta_min = float(delta_min)

    delta_max = request.form.get('delta_max', None)
    if delta_max:
        delta_max = float(delta_max)    

    epsilon = request.form.get('e', None)
    if epsilon:
        epsilon = float(epsilon)

    beta_value = request.form.get('beta', None)
    if beta_value:
        beta_value = float(beta_value)

    # Add delta_disclosure parameter
    delta_disclosure = request.form.get('delta_disclosure', None)
    if delta_disclosure:
        delta_disclosure = float(delta_disclosure)

    # Add p-sensitivity parameter
    p_value = request.form.get('p', None)
    if p_value:
        p_value = int(p_value)

    # Add c-value parameter for (c,k)-Safety
    c_value = request.form.get('c', None)
    if c_value:
        c_value = float(c_value)

    # Add delta parameter for differential privacy
    delta = request.form.get('delta', None)
    if delta:
        delta = float(delta)
    else:
        delta = 1e-6  # Default delta value for differential privacy

    # Add budget parameter for differential privacy
    budget = request.form.get('budget', None)
    if budget:
        budget = float(budget)
    else:
        budget = 1.0  # Default budget value (100%)

    quasi_identifiers = request.form.get('quasi_identifiers', 'Gender,Age,Zipcode').split(',')
    sensitive_column = request.form.get('sensitive_column', 'Disease').split(',')
    
    hierarchy_rules = json.loads(request.form.get('hierarchy_rules', '{}'))
    
    file_path = os.path.join(bp.root_path, 'uploads', file.filename)
    suppression_threshold = float(request.form.get('suppression_threshold', '0.3'))  # 抑制阈值
    file.save(file_path)
    dataPd = read_file(file_path)

    for qi in quasi_identifiers:
        if qi not in dataPd.columns:
            return jsonify({'error': f"Column '{qi}' does not exist in the file."}), 400

    try:
        if privacy_model == 'k-anonymity':
            resultPd = apply_k_anonymity_r(dataPd, quasi_identifiers, k_value, hierarchy_rules,suppression_threshold)
            if resultPd is None:
                raise ValueError("Anonymization process returned no result.")
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'l-diversity':
            resultPd = apply_l_diversity_r(dataPd, quasi_identifiers, sensitive_column, k_value,l_value, hierarchy_rules,suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply l-diversity.'}), 400
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 't-closeness':
            resultPd = apply_t_closeness_r(dataPd, quasi_identifiers, sensitive_column, k_value, t_value, hierarchy_rules,suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply t-closeness.'}), 400
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'km-anonymity':
            resultPd = apply_km_anonymity_r(dataPd, quasi_identifiers, sensitive_column,k_value, m_value, hierarchy_rules,suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply km-closeness.'}), 400
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'delta-presence':
            if delta_min is None or delta_max is None:
                return jsonify({'error': 'Both delta_min and delta_max are required for delta-presence'}), 400
            resultPd = apply_delta_presence(dataPd, quasi_identifiers, sensitive_column[0], delta_min, delta_max, hierarchy_rules, suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply delta-presence.'}), 400
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'beta-likeness':
            if beta_value is None:
                return jsonify({'error': 'Beta value is required for beta-likeness'}), 400
            resultPd = apply_beta_likeness_r(dataPd, quasi_identifiers, sensitive_column, beta_value, hierarchy_rules, suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply beta-likeness.'}), 400
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'delta-disclosure':
            if delta_disclosure is None:
                return jsonify({'error': 'Delta value is required for δ-disclosure privacy'}), 400
            resultPd = apply_delta_disclosure(dataPd, quasi_identifiers, sensitive_column, delta_disclosure, hierarchy_rules, suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply δ-disclosure privacy.'}), 400
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'p-sensitivity':
            if p_value is None:
                return jsonify({'error': 'P value is required for p-sensitivity'}), 400
            resultPd = apply_p_sensitivity(dataPd, quasi_identifiers, sensitive_column, p_value, hierarchy_rules, suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply p-sensitivity.'}), 400
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'ck-safety':
            if c_value is None:
                return jsonify({'error': 'C value is required for (c,k)-Safety'}), 400
            resultPd = apply_ck_safety(dataPd, quasi_identifiers, sensitive_column, 
                                     c_value, k_value, hierarchy_rules, suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply (c,k)-Safety.'}), 400
            return resultPd.to_json(orient='records')
        
        elif privacy_model == 'differential_privacy':
            if not epsilon:
                return jsonify({'error': 'Epsilon value is required for differential privacy'}), 400
            
            resultPd = apply_differential_privacy(dataPd, epsilon, delta, quasi_identifiers, 
                                                sensitive_column, hierarchy_rules, budget, suppression_threshold)
            if resultPd is None:
                return jsonify({'error': 'Failed to apply differential privacy.'}), 400
            
            return resultPd.to_json(orient='records')
        
        else:
            return jsonify({'error': f"Unsupported privacy model: {privacy_model}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unable to convert: {str(e)}"}), 400
