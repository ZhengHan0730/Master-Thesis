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
from scipy.spatial.distance import jensenshannon
from collections import Counter
from sklearn.metrics import mutual_info_score
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import time


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
# 对于文本类型的字段的统计方法js_divergence
def calculate_js_divergence(original_df, anonymized_df, columns_to_compare):
    results = []

    for col in columns_to_compare:
        try:
            orig_series = original_df[col].dropna().astype(str)
            anon_series = anonymized_df[col].dropna().astype(str)

            # 计算词频分布
            orig_counts = Counter(orig_series)
            anon_counts = Counter(anon_series)

            all_keys = set(orig_counts.keys()).union(set(anon_counts.keys()))
            orig_dist = np.array([orig_counts.get(k, 0) for k in all_keys], dtype=float)
            anon_dist = np.array([anon_counts.get(k, 0) for k in all_keys], dtype=float)

            if orig_dist.sum() == 0 or anon_dist.sum() == 0:
                results.append({
                    'column': col,
                    'metric': 'js-divergence',
                    'original': None,
                    'anonymized': None,
                    'difference': None,
                    'error': '分布为空，无法计算 JS 散度'
                })
                continue

            # 归一化为概率分布
            orig_dist /= orig_dist.sum()
            anon_dist /= anon_dist.sum()

            # 计算 Jensen-Shannon Divergence
            jsd = jensenshannon(orig_dist, anon_dist) ** 2

            results.append({
                'column': col,
                'metric': 'js-divergence',
                'original': None,
                'anonymized': None,
                'difference': round(jsd, 6)
            })

        except Exception as e:
            results.append({
                'column': col,
                'metric': 'js-divergence',
                'original': None,
                'anonymized': None,
                'difference': None,
                'error': str(e)
            })

    return results

# 针对文本信息：mutual_information
def calculate_mutual_information(original_df, anonymized_df, columns_to_compare):
    results = []

    for col in columns_to_compare:
        try:
            orig_series = original_df[col].dropna().astype(str)
            anon_series = anonymized_df[col].dropna().astype(str)

            # 对齐索引（仅对同一位置数据计算）
            min_len = min(len(orig_series), len(anon_series))
            orig_series = orig_series.iloc[:min_len]
            anon_series = anon_series.iloc[:min_len]

            joint_counter = Counter(zip(orig_series, anon_series))
            total = sum(joint_counter.values())

            if total == 0:
                results.append({
                    'column': col,
                    'metric': 'mutual-information',
                    'original': None,
                    'anonymized': None,
                    'difference': None,
                    'error': '无共同数据，无法计算'
                })
                continue

            px = Counter(orig_series)
            py = Counter(anon_series)

            mi = 0.0
            for (x, y), pxy in joint_counter.items():
                p_x = px[x]
                p_y = py[y]
                mi += (pxy / total) * log2((pxy * total) / (p_x * p_y))

            results.append({
                'column': col,
                'metric': 'mutual-information',
                'original': None,
                'anonymized': None,
                'difference': round(mi, 6)
            })

        except Exception as e:
            results.append({
                'column': col,
                'metric': 'mutual-information',
                'original': None,
                'anonymized': None,
                'difference': None,
                'error': str(e)
            })

    return results

# random forest
def evaluate_random_forest_quality(original_train_df, original_test_df, anonymized_train_df, anonymized_test_df, feature_columns, label_column):
    """
    用 Random Forest 对比原始数据和匿名数据的预测性能变化。
    使用用户提供的训练集和测试集进行评估。

    输入：
    - original_train_df: 原始训练集DataFrame
    - original_test_df: 原始测试集DataFrame
    - anonymized_train_df: 匿名化训练集DataFrame
    - anonymized_test_df: 匿名化测试集DataFrame
    - feature_columns: 参与建模的特征列列表
    - label_column: 目标列（标签）

    输出：
    - results: 列表，包含原始和匿名数据的Random Forest评估指标
    """

    results = []

    if not label_column:
        raise ValueError('必须提供 label 列用于 supervised learning')

    # 检查标签列是否存在于所有数据集中
    for df_name, df in [('原始训练集', original_train_df), ('原始测试集', original_test_df), 
                        ('匿名化训练集', anonymized_train_df), ('匿名化测试集', anonymized_test_df)]:
        if label_column not in df.columns:
            raise ValueError(f"{df_name}中不存在标签列 {label_column}")

    # 准备原始数据的特征和标签
    X_orig_train = original_train_df[feature_columns]
    y_orig_train = original_train_df[label_column]
    X_orig_test = original_test_df[feature_columns]
    y_orig_test = original_test_df[label_column]

    # 准备匿名数据的特征和标签
    X_anon_train = anonymized_train_df[feature_columns]
    y_anon_train = anonymized_train_df[label_column]
    X_anon_test = anonymized_test_df[feature_columns]
    y_anon_test = anonymized_test_df[label_column]

    # one-hot编码
    X_orig_train = pd.get_dummies(X_orig_train)
    X_orig_test = pd.get_dummies(X_orig_test)
    X_anon_train = pd.get_dummies(X_anon_train)
    X_anon_test = pd.get_dummies(X_anon_test)

    # 确保所有数据集具有相同的特征列
    all_columns = set()
    for df in [X_orig_train, X_orig_test, X_anon_train, X_anon_test]:
        all_columns.update(df.columns)
    
    # 为缺失列添加零值列
    for df in [X_orig_train, X_orig_test, X_anon_train, X_anon_test]:
        for col in all_columns:
            if col not in df.columns:
                df[col] = 0
    
    # 确保列顺序一致
    common_cols = sorted(list(all_columns))
    X_orig_train = X_orig_train[common_cols]
    X_orig_test = X_orig_test[common_cols]
    X_anon_train = X_anon_train[common_cols]
    X_anon_test = X_anon_test[common_cols]

    # 训练和评估原始数据模型
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(X_orig_train, y_orig_train)
    y_pred_orig = clf.predict(X_orig_test)

    results.append({
        'metric': 'random-forest',
        'dataset': 'Original',
        'accuracy': round(accuracy_score(y_orig_test, y_pred_orig), 4),
        'f1_score': round(f1_score(y_orig_test, y_pred_orig, average='macro'), 4),
        'precision': round(precision_score(y_orig_test, y_pred_orig, average='macro'), 4)
    })

    # 训练和评估匿名数据模型
    clf_an = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf_an.fit(X_anon_train, y_anon_train)
    y_pred_an = clf_an.predict(X_anon_test)

    results.append({
        'metric': 'random-forest',
        'dataset': 'Anonymized',
        'accuracy': round(accuracy_score(y_anon_test, y_pred_an), 4),
        'f1_score': round(f1_score(y_anon_test, y_pred_an, average='macro'), 4),
        'precision': round(precision_score(y_anon_test, y_pred_an, average='macro'), 4)
    })

    return results

# svm
def evaluate_svm_quality(original_train_df, original_test_df, anonymized_train_df, anonymized_test_df, feature_columns, label_column):
    """
    用优化的线性SVM对比原始数据和匿名数据的预测性能变化。
    使用用户提供的训练集和测试集进行评估。
    
    输入：
    - original_train_df: 原始训练集DataFrame
    - original_test_df: 原始测试集DataFrame
    - anonymized_train_df: 匿名化训练集DataFrame
    - anonymized_test_df: 匿名化测试集DataFrame
    - feature_columns: 参与建模的特征列列表
    - label_column: 目标列（标签）
    """
    
    results = []
    print(f"SVM评估开始，处理原始训练集 {len(original_train_df)} 行，测试集 {len(original_test_df)} 行数据")
    start_time = time.time()

    if not label_column:
        raise ValueError('必须提供 label 列用于 supervised learning')

    # 检查标签列是否存在于所有数据集中
    for df_name, df in [('原始训练集', original_train_df), ('原始测试集', original_test_df), 
                        ('匿名化训练集', anonymized_train_df), ('匿名化测试集', anonymized_test_df)]:
        if label_column not in df.columns:
            raise ValueError(f"{df_name}中不存在标签列 {label_column}")

    # 准备原始数据的特征和标签
    X_orig_train = original_train_df[feature_columns].copy()
    y_orig_train = original_train_df[label_column].copy()
    X_orig_test = original_test_df[feature_columns].copy()
    y_orig_test = original_test_df[label_column].copy()
    
    # 处理缺失的标签值
    valid_labels_mask_train = ~y_orig_train.isna()
    X_orig_train = X_orig_train[valid_labels_mask_train]
    y_orig_train = y_orig_train[valid_labels_mask_train]
    
    valid_labels_mask_test = ~y_orig_test.isna()
    X_orig_test = X_orig_test[valid_labels_mask_test]
    y_orig_test = y_orig_test[valid_labels_mask_test]
    
    # 进行one-hot编码
    X_orig_train = pd.get_dummies(X_orig_train)
    X_orig_test = pd.get_dummies(X_orig_test)
    
    # 准备匿名数据的特征和标签
    X_anon_train = anonymized_train_df[feature_columns].copy()
    y_anon_train = anonymized_train_df[label_column].copy()
    X_anon_test = anonymized_test_df[feature_columns].copy()
    y_anon_test = anonymized_test_df[label_column].copy()
    
    valid_labels_mask_anon_train = ~y_anon_train.isna()
    X_anon_train = X_anon_train[valid_labels_mask_anon_train]
    y_anon_train = y_anon_train[valid_labels_mask_anon_train]
    
    valid_labels_mask_anon_test = ~y_anon_test.isna()
    X_anon_test = X_anon_test[valid_labels_mask_anon_test]
    y_anon_test = y_anon_test[valid_labels_mask_anon_test]
    
    X_anon_train = pd.get_dummies(X_anon_train)
    X_anon_test = pd.get_dummies(X_anon_test)

    # 确保所有数据集具有相同的特征列
    all_columns = set()
    for df in [X_orig_train, X_orig_test, X_anon_train, X_anon_test]:
        all_columns.update(df.columns)
    
    # 为缺失列添加零值列
    for df in [X_orig_train, X_orig_test, X_anon_train, X_anon_test]:
        for col in all_columns:
            if col not in df.columns:
                df[col] = 0
    
    # 确保列顺序一致
    common_cols = sorted(list(all_columns))
    X_orig_train = X_orig_train[common_cols]
    X_orig_test = X_orig_test[common_cols]
    X_anon_train = X_anon_train[common_cols]
    X_anon_test = X_anon_test[common_cols]
    
    # 创建高效的SVM管道
    base_svc = LinearSVC(
        dual=False,     # 当样本数>特征数时更快
        C=1.0,
        max_iter=3000,
        tol=1e-4,
        random_state=42
    )
    
    # 使用管道进行预处理和模型训练
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svm', base_svc)
    ])
    
    print(f"开始训练原始数据SVM模型... ({time.time() - start_time:.1f}秒)")
    pipeline.fit(X_orig_train, y_orig_train)
    y_pred_orig = pipeline.predict(X_orig_test)
    
    accuracy = accuracy_score(y_orig_test, y_pred_orig)
    f1 = f1_score(y_orig_test, y_pred_orig, average='macro')
    precision = precision_score(y_orig_test, y_pred_orig, average='macro')
    
    results.append({
        'metric': 'svm',
        'dataset': 'Original',
        'accuracy': round(accuracy, 4),
        'f1_score': round(f1, 4),
        'precision': round(precision, 4)
    })
    
    print(f"原始数据评估完成，准确率: {accuracy:.4f} ({time.time() - start_time:.1f}秒)")

    # 匿名数据使用新的管道实例
    pipeline_anon = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(dual=False, C=1.0, max_iter=3000, tol=1e-4, random_state=42))
    ])
    
    print(f"开始训练匿名数据SVM模型... ({time.time() - start_time:.1f}秒)")
    pipeline_anon.fit(X_anon_train, y_anon_train)
    y_pred_an = pipeline_anon.predict(X_anon_test)
    
    accuracy_an = accuracy_score(y_anon_test, y_pred_an)
    f1_an = f1_score(y_anon_test, y_pred_an, average='macro')
    precision_an = precision_score(y_anon_test, y_pred_an, average='macro')
    
    results.append({
        'metric': 'svm',
        'dataset': 'Anonymized',
        'accuracy': round(accuracy_an, 4),
        'f1_score': round(f1_an, 4),
        'precision': round(precision_an, 4)
    })
    
    total_time = time.time() - start_time
    print(f"SVM评估完成，总耗时: {total_time:.1f}秒")

    return results

# MLP
def evaluate_mlp_quality(original_train_df, original_test_df, anonymized_train_df, anonymized_test_df, feature_columns, label_column):
    """
    使用多层感知器(MLP)神经网络对比原始数据和匿名数据的预测性能变化。
    使用用户提供的训练集和测试集进行评估。

    输入：
    - original_train_df: 原始训练集DataFrame
    - original_test_df: 原始测试集DataFrame
    - anonymized_train_df: 匿名化训练集DataFrame
    - anonymized_test_df: 匿名化测试集DataFrame
    - feature_columns: 参与建模的特征列列表
    - label_column: 目标列（标签）

    输出：
    - results: 列表，包含原始和匿名数据的MLP评估指标
    """
    
    results = []
    print(f"MLP评估开始，处理原始训练集 {len(original_train_df)} 行，测试集 {len(original_test_df)} 行数据")
    start_time = time.time()

    if not label_column:
        raise ValueError('必须提供 label 列用于 supervised learning')

    # 检查标签列是否存在于所有数据集中
    for df_name, df in [('原始训练集', original_train_df), ('原始测试集', original_test_df), 
                        ('匿名化训练集', anonymized_train_df), ('匿名化测试集', anonymized_test_df)]:
        if label_column not in df.columns:
            raise ValueError(f"{df_name}中不存在标签列 {label_column}")

    # 定义一个帮助函数处理数据集
    def prepare_dataset(df, feature_cols, label_col):
        try:
            # 复制数据避免修改原始数据
            df_copy = df.copy()
            
            # 处理标签列
            y = df_copy[label_col].copy()
            
            # 使用LabelEncoder处理分类标签
            if y.dtype == 'object' or y.dtype.name == 'category':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y.astype(str))
            else:
                # 尝试转换为数值类型
                y = pd.to_numeric(y, errors='coerce')
            
            # 移除标签中的NaN值
            valid_indices = ~pd.isna(y)
            if not valid_indices.all():
                print(f"发现并移除了 {(~valid_indices).sum()} 行带有NaN标签的数据")
                df_copy = df_copy.loc[valid_indices]
                y = y[valid_indices]
            
            # 特征数据处理
            X = df_copy[feature_cols].copy()
            
            # 检查并转换特征数据类型
            for col in X.columns:
                # 对非数值列进行特殊处理
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    # 检查是否可能是数值型
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except:
                        # 如果无法转换为数值，则将其转换为分类编码
                        X[col] = X[col].astype(str).fillna('missing').factorize()[0]
                        print(f"列 {col} 转换为分类编码")
            
            # 处理潜在的NaN值（转换为数值后可能产生）
            X = X.fillna(X.mean()).fillna(0)
            
            # 进行one-hot编码
            X = pd.get_dummies(X)
            
            return X, y
        
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            raise
    
    # 处理原始数据
    try:
        X_orig_train, y_orig_train = prepare_dataset(original_train_df, feature_columns, label_column)
        X_orig_test, y_orig_test = prepare_dataset(original_test_df, feature_columns, label_column)
        print(f"原始训练集特征形状: {X_orig_train.shape}, 标签形状: {y_orig_train.shape}")
        print(f"原始测试集特征形状: {X_orig_test.shape}, 标签形状: {y_orig_test.shape}")
    except Exception as e:
        print(f"原始数据预处理错误: {str(e)}")
        results.append({
            'metric': 'mlp',
            'dataset': 'Original',
            'accuracy': None,
            'f1_score': None,
            'precision': None,
            'error': str(e)
        })
        return results
    
    # 处理匿名数据
    try:
        X_anon_train, y_anon_train = prepare_dataset(anonymized_train_df, feature_columns, label_column)
        X_anon_test, y_anon_test = prepare_dataset(anonymized_test_df, feature_columns, label_column)
        print(f"匿名训练集特征形状: {X_anon_train.shape}, 标签形状: {y_anon_train.shape}")
        print(f"匿名测试集特征形状: {X_anon_test.shape}, 标签形状: {y_anon_test.shape}")
    except Exception as e:
        print(f"匿名数据预处理错误: {str(e)}")
        results.append({
            'metric': 'mlp',
            'dataset': 'Anonymized',
            'accuracy': None,
            'f1_score': None,
            'precision': None,
            'error': str(e)
        })
        # 继续评估原始数据
    
    # 确保所有数据集具有相同的特征列
    try:
        all_columns = set()
        for df in [X_orig_train, X_orig_test, X_anon_train, X_anon_test]:
            all_columns.update(df.columns)
        
        # 为缺失列添加零值列
        for df in [X_orig_train, X_orig_test, X_anon_train, X_anon_test]:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = 0
        
        # 确保列顺序一致
        common_cols = sorted(list(all_columns))
        X_orig_train = X_orig_train[common_cols]
        X_orig_test = X_orig_test[common_cols]
        X_anon_train = X_anon_train[common_cols]
        X_anon_test = X_anon_test[common_cols]
        
        print(f"特征列数量: {len(common_cols)}")
    except Exception as e:
        print(f"特征列对齐处理错误: {str(e)}")
        results.append({
            'metric': 'mlp',
            'dataset': 'Both',
            'accuracy': None,
            'f1_score': None,
            'precision': None,
            'error': str(e)
        })
        return results

    # 评估原始数据
    try:
        # 构建MLP分类器，配置合适的参数
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # 两个隐藏层
            activation='relu',              
            solver='adam',                  
            alpha=0.0001,                   # L2正则化
            batch_size='auto',              
            learning_rate='adaptive',       
            max_iter=300,                   
            early_stopping=True,            
            validation_fraction=0.1,        
            n_iter_no_change=10,            
            random_state=42,
            verbose=False  # 设为True可以查看训练过程
        )
        
        # 使用管道处理缺失值和标准化
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('mlp', mlp)
        ])
        
        print(f"开始训练原始数据MLP模型... ({time.time() - start_time:.1f}秒)")
        pipeline.fit(X_orig_train, y_orig_train)
        y_pred_orig = pipeline.predict(X_orig_test)
        
        # 确保计算指标时数据类型兼容
        accuracy = accuracy_score(y_orig_test, y_pred_orig)
        
        # 检查y_test的类型是否适合计算F1和精确度
        if len(np.unique(y_orig_test)) < 2:
            print("警告: 测试集中类别数少于2，无法正确计算F1和精确度")
            f1 = precision = 0.0
        else:
            f1 = f1_score(y_orig_test, y_pred_orig, average='macro', zero_division=0)
            precision = precision_score(y_orig_test, y_pred_orig, average='macro', zero_division=0)
        
        results.append({
            'metric': 'mlp',
            'dataset': 'Original',
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1, 4),
            'precision': round(precision, 4)
        })
        
        print(f"原始数据评估完成，准确率: {accuracy:.4f} ({time.time() - start_time:.1f}秒)")
    
    except Exception as e:
        print(f"原始数据MLP评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append({
            'metric': 'mlp',
            'dataset': 'Original',
            'accuracy': None,
            'f1_score': None,
            'precision': None,
            'error': str(e)
        })

    # 评估匿名数据
    try:
        # 匿名数据使用新的管道实例
        pipeline_anon = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42
            ))
        ])
        
        print(f"开始训练匿名数据MLP模型... ({time.time() - start_time:.1f}秒)")
        pipeline_anon.fit(X_anon_train, y_anon_train)
        y_pred_an = pipeline_anon.predict(X_anon_test)
        
        # 同样确保计算指标时数据类型兼容
        accuracy_an = accuracy_score(y_anon_test, y_pred_an)
        
        if len(np.unique(y_anon_test)) < 2:
            print("警告: 匿名测试集中类别数少于2，无法正确计算F1和精确度")
            f1_an = precision_an = 0.0
        else:
            f1_an = f1_score(y_anon_test, y_pred_an, average='macro', zero_division=0)
            precision_an = precision_score(y_anon_test, y_pred_an, average='macro', zero_division=0)
        
        results.append({
            'metric': 'mlp',
            'dataset': 'Anonymized',
            'accuracy': round(accuracy_an, 4),
            'f1_score': round(f1_an, 4),
            'precision': round(precision_an, 4)
        })
        
        print(f"匿名数据评估完成，准确率: {accuracy_an:.4f} ({time.time() - start_time:.1f}秒)")
    
    except Exception as e:
        print(f"匿名数据MLP评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append({
            'metric': 'mlp',
            'dataset': 'Anonymized',
            'accuracy': None,
            'f1_score': None,
            'precision': None,
            'error': str(e)
        })
    
    total_time = time.time() - start_time
    print(f"MLP评估完成，总耗时: {total_time:.1f}秒")

    return results


# knn
def evaluate_unsupervised_quality(original_df, anonymized_df, feature_columns, k_neighbors=5, n_clusters=5):
    """
    无监督学习方法综合评估：
    - KMeans聚类 + Silhouette得分
    - KNN邻居保持率
    - LOF局部异常因子变化

    参数：
    - original_df: 原始数据 DataFrame
    - anonymized_df: 匿名化后的 DataFrame
    - feature_columns: 特征列
    - k_neighbors: KNN邻居数（默认5）
    - n_clusters: KMeans聚类簇数（默认5）

    返回：
    - results: 评估结果列表
    """

    results = []

    # 只取特征列，进行独热编码 (One-Hot Encoding)
    X_orig = pd.get_dummies(original_df[feature_columns].copy())
    X_anon = pd.get_dummies(anonymized_df[feature_columns].copy())

    # 取两边共有的列，避免匿名化时特征值变化导致问题
    common_cols = list(set(X_orig.columns) & set(X_anon.columns))
    if not common_cols:
        raise ValueError("原始数据和匿名数据没有共同特征，无法比较！")

    X_orig = X_orig[common_cols]
    X_anon = X_anon[common_cols]

    ###### 1. KMeans聚类 + Silhouette得分 ######
    try:
        kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_orig)
        kmeans_anon = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_anon)

        sil_orig = silhouette_score(X_orig, kmeans_orig.labels_)
        sil_anon = silhouette_score(X_anon, kmeans_anon.labels_)

        results.append({
            'metric': 'kmeans_silhouette',
            'original': round(sil_orig, 6),
            'anonymized': round(sil_anon, 6),
            'difference': round(abs(sil_orig - sil_anon), 6)
        })
    except Exception as e:
        results.append({
            'metric': 'kmeans_silhouette',
            'error': str(e)
        })

    ###### 2. 邻居保持率 (Neighbor Preservation) ######
    try:
        knn_orig = NearestNeighbors(n_neighbors=k_neighbors).fit(X_orig)
        _, indices_orig = knn_orig.kneighbors(X_orig)

        knn_anon = NearestNeighbors(n_neighbors=k_neighbors).fit(X_anon)
        _, indices_anon = knn_anon.kneighbors(X_anon)

        preserve_count = 0
        total_count = 0

        for idx in range(min(len(indices_orig), len(indices_anon))):
            neighbors_orig = set(indices_orig[idx][1:])  # 排除自己
            neighbors_anon = set(indices_anon[idx][1:])
            preserve_count += len(neighbors_orig & neighbors_anon)
            total_count += len(neighbors_orig)

        preservation_ratio = preserve_count / total_count if total_count else 0

        results.append({
            'metric': 'knn_neighbor_preservation',
            'original': 1.0,
            'anonymized': round(preservation_ratio, 6),
            'difference': round(1.0 - preservation_ratio, 6)
        })
    except Exception as e:
        results.append({
            'metric': 'knn_neighbor_preservation',
            'error': str(e)
        })

    ###### 3. LOF局部异常因子变化 ######
    try:
        lof_orig = LocalOutlierFactor(n_neighbors=k_neighbors)
        lof_score_orig = lof_orig.fit_predict(X_orig)
        mean_lof_orig = np.mean(-lof_score_orig)

        lof_anon = LocalOutlierFactor(n_neighbors=k_neighbors)
        lof_score_anon = lof_anon.fit_predict(X_anon)
        mean_lof_anon = np.mean(-lof_score_anon)

        results.append({
            'metric': 'local_outlier_factor',
            'original': round(mean_lof_orig, 6),
            'anonymized': round(mean_lof_anon, 6),
            'difference': round(abs(mean_lof_orig - mean_lof_anon), 6)
        })
    except Exception as e:
        results.append({
            'metric': 'local_outlier_factor',
            'error': str(e)
        })

    return results


# 区间值预处理
def parse_range(value, default_value=0):
    """处理各种不同格式的区间值，返回中点值，避免返回NaN"""
    # 如果是None或不是字符串，尝试直接转为浮点数
    if not isinstance(value, str):
        try:
            return float(value)
        except:
            return default_value  # 返回默认值而不是NaN
    
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
            clean_value = value.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            start, end = map(float, clean_value.split(','))
            return (start + end) / 2
        except:
            pass
    
    # 尝试直接转换为浮点数
    try:
        return float(value)
    except:
        return default_value  # 返回默认值而不是NaN

# 新增：用于将原始日期或匿名化日期区间转为距今天数
def convert_date_to_days(column, is_range=False, default_days=0):
    """
    将日期列转换为距今的天数，使用默认值避免空值
    """
    today = datetime.today()

    if is_range:
        def parse_date_range(date_str):
            try:
                if pd.isna(date_str):
                    return default_days
                start_str, end_str = date_str.split(' - ')
                start = pd.to_datetime(start_str.strip(), errors='coerce')
                end = pd.to_datetime(end_str.strip(), errors='coerce')
                if pd.isna(start) or pd.isna(end):
                    return default_days
                midpoint = start + (end - start) / 2
                return (today - midpoint).days
            except Exception as e:
                print(f"日期区间解析失败: {date_str}，错误: {e}")
                return default_days

        return column.apply(parse_date_range)

    else:
        # 使用fillna处理转换后的空值
        dates = pd.to_datetime(column, errors='coerce')
        days = (today - dates).dt.days
        return days.fillna(default_days)
   

# 对列进行预处理
def preprocess_column(column, default_numeric=0, default_date=0):
    """
    自动识别并处理列数据，避免返回空值
    """
    # 先处理完全空的列
    if column.dropna().empty:
        return pd.Series([default_numeric] * len(column))
    
    sample_value = column.dropna().astype(str).iloc[0]

    # 简单规则判断是否为日期或日期区间
    if any(ch in sample_value for ch in ['-', '/']) and any(kw in sample_value for kw in ['20', '19']):
        is_range = ' - ' in sample_value
        return convert_date_to_days(column, is_range=is_range, default_days=default_date)

    # 其余默认按数值区间处理
    return column.apply(lambda x: parse_range(x, default_value=default_numeric))

@bp.route('/evaluation', methods=['POST']) 
def data_quality_evaluation():
    # 获取用户选择的指标
    metrics = request.form.get('metrics', 'mean,median,variance').split(',')
    
    # 确定是否包含机器学习评估方法
    ml_metrics = ['random-forest', 'svm', 'mlp']
    is_ml_evaluation = any(m in ml_metrics for m in metrics)
    
    # 区分数值型统计方法和文本型统计方法
    numeric_metrics = ['mean', 'median', 'variance', 'wasserstein', 'ks_similarity', 'pearson', 'spearman']
    text_metrics = ['js-divergence', 'mutual-information']
    unsupervised_metrics = ['unsupervised-quality']
    
    # 根据评估类型确定所需文件
    if is_ml_evaluation:
        # 机器学习评估需要四个文件
        required_files = ['original_train_file', 'original_test_file', 
                          'anonymized_train_file', 'anonymized_test_file']
        
        # 检查是否提供了所有必要文件
        for file_key in required_files:
            if file_key not in request.files:
                return jsonify({'error': f'机器学习评估方法需要提供四个文件: 原始训练集、原始测试集、匿名化训练集和匿名化测试集。缺少 {file_key}'}), 400
            
        # 获取文件对象
        original_train_file = request.files['original_train_file']
        original_test_file = request.files['original_test_file']
        anonymized_train_file = request.files['anonymized_train_file']
        anonymized_test_file = request.files['anonymized_test_file']
        
        # 检查文件扩展名
        for file in [original_train_file, original_test_file, anonymized_train_file, anonymized_test_file]:
            if not allowed_file(file.filename):
                return jsonify({'error': '只支持 .csv 或 .tsv 文件格式'}), 400
        
        # 读取文件
        try:
            original_train_df = pd.read_csv(original_train_file, on_bad_lines='skip')
            original_test_df = pd.read_csv(original_test_file, on_bad_lines='skip')
            anonymized_train_df = pd.read_csv(anonymized_train_file, on_bad_lines='skip')
            anonymized_test_df = pd.read_csv(anonymized_test_file, on_bad_lines='skip')
            
            # 合并训练和测试集，用于非ML评估
            original_df = pd.concat([original_train_df, original_test_df], ignore_index=True)
            anonymized_df = pd.concat([anonymized_train_df, anonymized_test_df], ignore_index=True)
        except Exception as e:
            return jsonify({'error': f'文件读取失败，请检查格式：{str(e)}'}), 400
    else:
        # 非机器学习评估只需要两个文件
        if 'original_file' not in request.files or 'anonymized_file' not in request.files:
            return jsonify({'error': '必须同时上传原始文件和匿名化文件'}), 400
        
        original_file = request.files['original_file']
        anonymized_file = request.files['anonymized_file']
        
        # 检查文件扩展名
        if not allowed_file(original_file.filename) or not allowed_file(anonymized_file.filename):
            return jsonify({'error': '只支持 .csv 或 .tsv 文件格式'}), 400
        
        # 读取文件
        try:
            original_df = pd.read_csv(original_file, on_bad_lines='skip')
            anonymized_df = pd.read_csv(anonymized_file, on_bad_lines='skip')
        except Exception as e:
            return jsonify({'error': f'文件读取失败，请检查格式：{str(e)}'}), 400
    
    columns_to_compare = request.form.get('columns')
    label_column = request.form.get('label')
    
    if not columns_to_compare:
        return jsonify({'error': '必须指定用于比较的列'}), 400
    
    columns_to_compare = columns_to_compare.split(',')
    
    # 检查列是否存在
    if is_ml_evaluation:
        # 检查所有四个数据集中的列
        for df_name, df in [('原始训练集', original_train_df), ('原始测试集', original_test_df), 
                            ('匿名化训练集', anonymized_train_df), ('匿名化测试集', anonymized_test_df)]:
            missing_cols = [col for col in columns_to_compare if col not in df.columns]
            if missing_cols:
                return jsonify({'error': f'{df_name}中以下列不存在: {missing_cols}'}), 400
            
            # 检查标签列
            if label_column and label_column not in df.columns:
                return jsonify({'error': f'{df_name}中标签列 {label_column} 不存在'}), 400
    else:
        # 只检查两个数据集中的列
        missing_cols = [col for col in columns_to_compare if col not in original_df.columns or col not in anonymized_df.columns]
        if missing_cols:
            return jsonify({'error': f'以下列在文件中不存在: {missing_cols}'}), 400
    
    # 如果选择了数值统计方法，先验证数值类型，对非数值类型尝试区间处理
    numeric_columns_to_compare = columns_to_compare.copy()
    
    if any(metric in metrics for metric in numeric_metrics + ml_metrics + unsupervised_metrics):
        # 收集非数值类型的列
        if is_ml_evaluation:
            # 检查所有四个数据集
            non_numeric_cols = []
            for df in [original_train_df, original_test_df, anonymized_train_df, anonymized_test_df]:
                for col in columns_to_compare:
                    if not pd.api.types.is_numeric_dtype(df[col]) and col not in non_numeric_cols:
                        non_numeric_cols.append(col)
        else:
            # 只检查两个数据集
            non_numeric_cols = [
                col for col in columns_to_compare
                if not pd.api.types.is_numeric_dtype(original_df[col]) or not pd.api.types.is_numeric_dtype(anonymized_df[col])
            ]
        
        # 对非数值类型的列尝试预处理
        for col in non_numeric_cols:
            try:
                # 生成处理后的列名
                processed_col_name = col + '_processed'
                
                # 根据评估类型处理相应的数据集
                if is_ml_evaluation:
                    # 处理四个数据集
                    original_train_df[processed_col_name] = preprocess_column(original_train_df[col])
                    original_train_df[processed_col_name] = pd.to_numeric(original_train_df[processed_col_name], errors='coerce').fillna(0.0).astype(float)
                    
                    original_test_df[processed_col_name] = preprocess_column(original_test_df[col])
                    original_test_df[processed_col_name] = pd.to_numeric(original_test_df[processed_col_name], errors='coerce').fillna(0.0).astype(float)
                    
                    anonymized_train_df[processed_col_name] = preprocess_column(anonymized_train_df[col])
                    anonymized_train_df[processed_col_name] = pd.to_numeric(anonymized_train_df[processed_col_name], errors='coerce').fillna(0.0).astype(float)
                    
                    anonymized_test_df[processed_col_name] = preprocess_column(anonymized_test_df[col])
                    anonymized_test_df[processed_col_name] = pd.to_numeric(anonymized_test_df[processed_col_name], errors='coerce').fillna(0.0).astype(float)
                else:
                    # 处理两个数据集
                    original_df[processed_col_name] = preprocess_column(original_df[col])
                    original_df[processed_col_name] = pd.to_numeric(original_df[processed_col_name], errors='coerce').fillna(0.0).astype(float)
                    
                    anonymized_df[processed_col_name] = preprocess_column(anonymized_df[col])
                    anonymized_df[processed_col_name] = pd.to_numeric(anonymized_df[processed_col_name], errors='coerce').fillna(0.0).astype(float)
                
                # 更新列名用于后续比较
                numeric_columns_to_compare = [processed_col_name if c == col else c for c in numeric_columns_to_compare]
            except Exception as e:
                return jsonify({'error': f'数据预处理失败: {col}, 错误: {str(e)}'}), 400
        
        # 再次验证是否有非数值类型的列
        if is_ml_evaluation:
            numeric_check_failed = []
            for df in [original_train_df, original_test_df, anonymized_train_df, anonymized_test_df]:
                for col in numeric_columns_to_compare:
                    if not pd.api.types.is_numeric_dtype(df[col]) and col not in numeric_check_failed:
                        numeric_check_failed.append(col)
        else:
            numeric_check_failed = [
                col for col in numeric_columns_to_compare
                if not pd.api.types.is_numeric_dtype(original_df[col]) or not pd.api.types.is_numeric_dtype(anonymized_df[col])
            ]
        
        if numeric_check_failed and any(metric in metrics for metric in numeric_metrics):
            return jsonify({'error': f'以下列不是数值类型，无法计算统计指标: {numeric_check_failed}'}), 400
    
    results = []
    
    # 处理普通统计指标（非ML指标）
    if any(m in metrics for m in numeric_metrics + text_metrics + unsupervised_metrics):
        # 处理数值型指标
        if 'mean' in metrics:
            results.extend(calculate_mean_diff(original_df, anonymized_df, numeric_columns_to_compare))
        if 'median' in metrics:
            results.extend(calculate_median_diff(original_df, anonymized_df, numeric_columns_to_compare))
        if 'variance' in metrics:
            results.extend(calculate_variance_diff(original_df, anonymized_df, numeric_columns_to_compare))
        if 'wasserstein' in metrics:
            results.extend(calculate_wasserstein_distance(original_df, anonymized_df, numeric_columns_to_compare))
        if 'ks_similarity' in metrics:
            results.extend(calculate_ks_similarity(original_df, anonymized_df, numeric_columns_to_compare))
        if 'pearson' in metrics:
            results.extend(calculate_pearson(original_df, anonymized_df, numeric_columns_to_compare))
        if 'spearman' in metrics:
            results.extend(calculate_spearman(original_df, anonymized_df, numeric_columns_to_compare))
    
        # 处理文本指标
        if any(m in text_metrics for m in metrics):
            if 'js-divergence' in metrics:
                results.extend(calculate_js_divergence(original_df, anonymized_df, columns_to_compare))
            if 'mutual-information' in metrics:
                results.extend(calculate_mutual_information(original_df, anonymized_df, columns_to_compare))
    
        # 处理无监督学习指标
        if any(m in unsupervised_metrics for m in metrics):
            try:
                results.extend(evaluate_unsupervised_quality(original_df, anonymized_df, feature_columns=numeric_columns_to_compare, k_neighbors=5, n_clusters=5))
            except Exception as e:
                return jsonify({'error': f'unsupervised-quality 评估失败: {str(e)}'}), 400
    
    # 处理机器学习评估指标
    # 在evaluation路由函数中处理机器学习评估部分
    if is_ml_evaluation:
        if 'random-forest' in metrics:
            try:
                results.extend(
                    evaluate_random_forest_quality(
                        original_train_df, original_test_df,
                        anonymized_train_df, anonymized_test_df,
                        feature_columns=numeric_columns_to_compare,
                        label_column=label_column
                    )
                )
            except Exception as e:
                return jsonify({'error': f'random-forest 评估失败: {str(e)}'}), 400
            
        if 'svm' in metrics:
            try:
                results.extend(
                    evaluate_svm_quality(
                        original_train_df, original_test_df,
                        anonymized_train_df, anonymized_test_df,
                        feature_columns=numeric_columns_to_compare,
                        label_column=label_column
                    )
                )
            except Exception as e:
                return jsonify({'error': f'SVM 评估失败: {str(e)}'}), 400
            
        if 'mlp' in metrics:
            try:
                results.extend(
                    evaluate_mlp_quality(
                        original_train_df, original_test_df,
                        anonymized_train_df, anonymized_test_df,
                        feature_columns=numeric_columns_to_compare,
                        label_column=label_column
                    )
                )
            except Exception as e:
                return jsonify({'error': f'MLP 评估失败: {str(e)}'}), 400
        
        # 生成结果ID并保存结果
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

    # 获取数据集划分比例
    train_ratio = request.form.get('train_ratio', None)
    if train_ratio:
        train_ratio = float(train_ratio)
        if train_ratio <= 0 or train_ratio >= 1:
            return jsonify({'error': 'Train ratio must be between 0 and 1'}), 400

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

    delta_disclosure = request.form.get('delta_disclosure', None)
    if delta_disclosure:
        delta_disclosure = float(delta_disclosure)

    p_value = request.form.get('p', None)
    if p_value:
        p_value = int(p_value)

    c_value = request.form.get('c', None)
    if c_value:
        c_value = float(c_value)

    delta = request.form.get('delta', None)
    if delta:
        delta = float(delta)
    else:
        delta = 1e-6  # Default delta value for differential privacy

    budget = request.form.get('budget', None)
    if budget:
        budget = float(budget)
    else:
        budget = 1.0  # Default budget value (100%)

    quasi_identifiers = request.form.get('quasi_identifiers', 'Gender,Age,Zipcode').split(',')
    sensitive_column = request.form.get('sensitive_column', 'Disease').split(',')
    
    hierarchy_rules = json.loads(request.form.get('hierarchy_rules', '{}'))
    
    file_path = os.path.join(bp.root_path, 'uploads', file.filename)
    output_dir = os.path.join(bp.root_path, 'results')
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取文件名（不包含扩展名）和扩展名
    file_name = os.path.splitext(file.filename)[0]
    file_ext = os.path.splitext(file.filename)[1]
    
    suppression_threshold = float(request.form.get('suppression_threshold', '0.3'))
    file.save(file_path)
    dataPd = read_file(file_path)

    for qi in quasi_identifiers:
        if qi not in dataPd.columns:
            return jsonify({'error': f"Column '{qi}' does not exist in the file."}), 400

    try:
        # 生成时间戳，用于确保文件名唯一
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_files = {}
        
        # 如果提供了训练集比例，则进行数据集划分
        if train_ratio:
            # 随机划分数据集
            train_data, test_data = train_test_split(dataPd, train_size=train_ratio, random_state=42)
            
            # 保存原始训练集和测试集
            original_train_file = f"{file_name}_original_train_{timestamp}{file_ext}"
            original_train_path = os.path.join(output_dir, original_train_file)
            save_file(train_data, original_train_path)
            result_files['original_train'] = original_train_file
            
            original_test_file = f"{file_name}_original_test_{timestamp}{file_ext}"
            original_test_path = os.path.join(output_dir, original_test_file)
            save_file(test_data, original_test_path)
            result_files['original_test'] = original_test_file
            
            # 对训练集进行匿名化处理
            train_anonymized = anonymize_data(train_data, privacy_model, quasi_identifiers, sensitive_column, 
                                             k_value, l_value, t_value, m_value, delta_min, delta_max, 
                                             epsilon, beta_value, delta_disclosure, p_value, c_value, 
                                             delta, budget, hierarchy_rules, suppression_threshold)
            
            if train_anonymized is None:
                return jsonify({'error': 'Failed to anonymize training data.'}), 400
            
            # 保存匿名化训练集
            anonymized_train_file = f"{file_name}_{privacy_model}_train_{timestamp}{file_ext}"
            anonymized_train_path = os.path.join(output_dir, anonymized_train_file)
            save_file(train_anonymized, anonymized_train_path)
            result_files['anonymized_train'] = anonymized_train_file
            
            # 对测试集进行匿名化处理
            test_anonymized = anonymize_data(test_data, privacy_model, quasi_identifiers, sensitive_column, 
                                           k_value, l_value, t_value, m_value, delta_min, delta_max, 
                                           epsilon, beta_value, delta_disclosure, p_value, c_value, 
                                           delta, budget, hierarchy_rules, suppression_threshold)
            
            if test_anonymized is None:
                return jsonify({'error': 'Failed to anonymize test data.'}), 400
            
            # 保存匿名化测试集
            anonymized_test_file = f"{file_name}_{privacy_model}_test_{timestamp}{file_ext}"
            anonymized_test_path = os.path.join(output_dir, anonymized_test_file)
            save_file(test_anonymized, anonymized_test_path)
            result_files['anonymized_test'] = anonymized_test_file
            
            # 返回包含文件路径的JSON结果
            return jsonify({
                'status': 'success',
                'files': result_files,
                'message': f'Files saved in results directory: {", ".join(result_files.values())}'
            })
        
        else:
            # 如果没有提供划分比例，则直接对整个数据集进行匿名化
            anonymized_data = anonymize_data(dataPd, privacy_model, quasi_identifiers, sensitive_column, 
                                           k_value, l_value, t_value, m_value, delta_min, delta_max, 
                                           epsilon, beta_value, delta_disclosure, p_value, c_value, 
                                           delta, budget, hierarchy_rules, suppression_threshold)
            
            if anonymized_data is None:
                return jsonify({'error': 'Failed to anonymize data.'}), 400
            
            # 保存匿名化数据
            anonymized_file = f"{file_name}_{privacy_model}_{timestamp}{file_ext}"
            anonymized_path = os.path.join(output_dir, anonymized_file)
            save_file(anonymized_data, anonymized_path)
            result_files['anonymized'] = anonymized_file
            
            # 返回包含文件路径的JSON结果
            return jsonify({
                'status': 'success',
                'files': result_files,
                'message': f'File saved in results directory: {anonymized_file}'
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Unable to process: {str(e)}"}), 400
    
# 添加匿名化文件下载接口
@bp.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    提供结果文件的下载
    
    Args:
        filename: 要下载的文件名
    """
    try:
        # 检查文件是否存在于results目录
        output_dir = os.path.join(bp.root_path, 'results')
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'File {filename} not found'}), 404
        
        # 返回文件
        return send_file(
            file_path, 
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

# 辅助函数：根据文件扩展名保存数据
def save_file(data, file_path):
    """
    根据文件扩展名保存数据到指定路径
    
    Args:
        data: 要保存的数据（DataFrame）
        file_path: 文件保存路径
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        data.to_csv(file_path, index=False)
    elif file_ext in ['.xls', '.xlsx']:
        data.to_excel(file_path, index=False)
    elif file_ext == '.json':
        data.to_json(file_path, orient='records')
    else:
        # 默认保存为CSV
        data.to_csv(file_path, index=False)



# 提取匿名化逻辑到独立函数，便于复用
def anonymize_data(data, privacy_model, quasi_identifiers, sensitive_column, k_value=None, l_value=None, 
                  t_value=None, m_value=None, delta_min=None, delta_max=None, epsilon=None, beta_value=None, 
                  delta_disclosure=None, p_value=None, c_value=None, delta=None, budget=None, 
                  hierarchy_rules=None, suppression_threshold=0.3):
    """
    根据指定的隐私模型对数据进行匿名化处理
    
    Args:
        data: 要匿名化的数据集
        privacy_model: 使用的隐私模型名称
        quasi_identifiers: 准标识符列表
        sensitive_column: 敏感属性列表
        其他参数: 各隐私模型所需的参数
        
    Returns:
        匿名化后的数据集或None（处理失败）
    """
    if privacy_model == 'k-anonymity':
        return apply_k_anonymity_r(data, quasi_identifiers, k_value, hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 'l-diversity':
        return apply_l_diversity_r(data, quasi_identifiers, sensitive_column, k_value, l_value, 
                                 hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 't-closeness':
        return apply_t_closeness_r(data, quasi_identifiers, sensitive_column, k_value, t_value, 
                                 hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 'km-anonymity':
        return apply_km_anonymity_r(data, quasi_identifiers, sensitive_column, k_value, m_value, 
                                  hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 'delta-presence':
        if delta_min is None or delta_max is None:
            return None
        return apply_delta_presence(data, quasi_identifiers, sensitive_column[0], delta_min, delta_max, 
                                  hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 'beta-likeness':
        if beta_value is None:
            return None
        return apply_beta_likeness_r(data, quasi_identifiers, sensitive_column, beta_value, 
                                   hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 'delta-disclosure':
        if delta_disclosure is None:
            return None
        return apply_delta_disclosure(data, quasi_identifiers, sensitive_column, delta_disclosure, 
                                    hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 'p-sensitivity':
        if p_value is None:
            return None
        return apply_p_sensitivity(data, quasi_identifiers, sensitive_column, p_value, 
                                 hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 'ck-safety':
        if c_value is None:
            return None
        return apply_ck_safety(data, quasi_identifiers, sensitive_column, c_value, k_value, 
                             hierarchy_rules, suppression_threshold)
    
    elif privacy_model == 'differential_privacy':
        if not epsilon:
            return None
        return apply_differential_privacy(data, epsilon, delta, quasi_identifiers, sensitive_column, 
                                        hierarchy_rules, budget, suppression_threshold)
    
    else:
        return None