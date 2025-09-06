# NIPT问题3：多因素综合风险评估与最优检测时点
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import traceback
import re

warnings.filterwarnings('ignore')

# 设置中文字体和美化样式
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免GUI冲突
import matplotlib.font_manager as fm

# 设置中文字体支持
import os
import matplotlib.pyplot as plt

# 强制设置matplotlib使用中文字体
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置plt的中文字体
plt.rcParams['font.family'] = ['sans-serif'] 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置字体编码
import locale
try:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese_China.936')
    except:
        pass

print("设置中文字体支持: Microsoft YaHei")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16
plt.style.use('default')  # 使用默认样式以确保兼容性
sns.set_palette("Set2")

class MultifactorNIPTOptimizer:
    """多因素NIPT优化器"""
    
    def __init__(self, data_file='附件.xlsx'):
        self.data_file = data_file
        self.threshold = 0.04
        self.alpha = 0.1  # 90%置信水平
        self.measurement_error_std = 0.002  # 检测误差标准差
        
    def load_and_process_data(self):
        """加载并处理数据"""
        print("="*80)
        print("多因素数据加载与预处理")
        print("="*80)
        
        try:
            # 读取原始数据
            original_data = pd.read_excel(self.data_file, sheet_name=0)
            print(f"数据加载成功，原始数据形状: {original_data.shape}")
            
            # 转换孕周格式
            def convert_gestation_week(week_str):
                if pd.isna(week_str):
                    return np.nan
                week_str = str(week_str).strip()
                pattern = r'(\d+)(?:w\+?(\d+))?|(\d+\.\d+)'
                match = re.search(pattern, week_str.lower())
                
                if match:
                    if match.group(2):  # 格式如 "13w+6"
                        weeks = float(match.group(1))
                        days = float(match.group(2))
                        return weeks + days / 7
                    elif match.group(3):  # 直接数值
                        return float(match.group(3))
                    else:  # 只有周数
                        return float(match.group(1))
                
                try:
                    return float(week_str)
                except ValueError:
                    return np.nan
            
            original_data['孕周_数值'] = original_data['检测孕周'].apply(convert_gestation_week)
            
            # 过滤男胎数据并添加必要的字段清洗
            male_data = original_data[original_data['Y染色体浓度'] > 0].copy()
            male_data = male_data.dropna(subset=['孕周_数值', '孕妇BMI', 'Y染色体浓度'])
            
            # 数据清洗和异常值处理
            print(f"清洗前数据量: {len(male_data)}")
            
            # BMI异常值处理
            male_data = male_data[(male_data['孕妇BMI'] >= 15) & (male_data['孕妇BMI'] <= 50)]
            
            # 年龄异常值处理
            if '年龄' in male_data.columns:
                male_data = male_data[(male_data['年龄'] >= 18) & (male_data['年龄'] <= 50)]
            
            # 身高体重异常值处理
            if '身高' in male_data.columns:
                male_data['身高'] = pd.to_numeric(male_data['身高'], errors='coerce')
                male_data = male_data[(male_data['身高'] >= 140) & (male_data['身高'] <= 200)]
            
            if '体重' in male_data.columns:
                male_data['体重'] = pd.to_numeric(male_data['体重'], errors='coerce')
                male_data = male_data[(male_data['体重'] >= 40) & (male_data['体重'] <= 120)]
            
            # 如果没有BMI但有身高体重，则计算BMI
            if '身高' in male_data.columns and '体重' in male_data.columns:
                missing_bmi = male_data['孕妇BMI'].isna()
                if missing_bmi.any():
                    male_data.loc[missing_bmi, '孕妇BMI'] = male_data.loc[missing_bmi, '体重'] / ((male_data.loc[missing_bmi, '身高'] / 100) ** 2)
            
            print(f"清洗后数据量: {len(male_data)}")
            
            # 构建多因素生存数据
            survival_data = []
            
            for woman_code in male_data['孕妇代码'].unique():
                woman_data = male_data[male_data['孕妇代码'] == woman_code].copy()
                woman_data = woman_data.sort_values('孕周_数值')
                
                if len(woman_data) == 0:
                    continue
                
                # 基础信息
                bmi = woman_data['孕妇BMI'].iloc[0]
                age = woman_data['年龄'].iloc[0] if '年龄' in woman_data.columns else np.nan
                height = woman_data['身高'].iloc[0] if '身高' in woman_data.columns else np.nan
                weight = woman_data['体重'].iloc[0] if '体重' in woman_data.columns else np.nan
                
                # 额外的生物学和临床特征
                pregnancy_count = pd.to_numeric(woman_data['怀孕次数'].iloc[0], errors='coerce') if '怀孕次数' in woman_data.columns else np.nan
                birth_count = pd.to_numeric(woman_data['生产次数'].iloc[0], errors='coerce') if '生产次数' in woman_data.columns else np.nan
                ivf_pregnancy = 1 if woman_data['IVF妊娠'].iloc[0] == 'IVF妊娠' else 0 if 'IVF妊娠' in woman_data.columns else np.nan
                
                # 检测技术相关特征
                avg_gc_content = pd.to_numeric(woman_data['GC含量'], errors='coerce').mean() if 'GC含量' in woman_data.columns else np.nan
                avg_mapping_rate = pd.to_numeric(woman_data['在参考基因组上比对的比例'], errors='coerce').mean() if '在参考基因组上比对的比例' in woman_data.columns else np.nan
                avg_duplicate_rate = pd.to_numeric(woman_data['重复读段的比例'], errors='coerce').mean() if '重复读段的比例' in woman_data.columns else np.nan
                avg_unique_reads = pd.to_numeric(woman_data['唯一比对的读段数'], errors='coerce').mean() if '唯一比对的读段数' in woman_data.columns else np.nan
                avg_filtered_rate = pd.to_numeric(woman_data['被过滤掉读段数的比例'], errors='coerce').mean() if '被过滤掉读段数的比例' in woman_data.columns else np.nan
                
                # 染色体异常相关特征
                avg_chr13_z = pd.to_numeric(woman_data['13号染色体的Z值'], errors='coerce').mean() if '13号染色体的Z值' in woman_data.columns else np.nan
                avg_chr18_z = pd.to_numeric(woman_data['18号染色体的Z值'], errors='coerce').mean() if '18号染色体的Z值' in woman_data.columns else np.nan
                avg_chr21_z = pd.to_numeric(woman_data['21号染色体的Z值'], errors='coerce').mean() if '21号染色体的Z值' in woman_data.columns else np.nan
                avg_x_concentration = pd.to_numeric(woman_data['X染色体浓度'], errors='coerce').mean() if 'X染色体浓度' in woman_data.columns else np.nan
                
                # 确定事件时间和删失状态
                reaching_records = woman_data[woman_data['Y染色体浓度'] >= self.threshold]
                
                if len(reaching_records) > 0:
                    event_time = reaching_records['孕周_数值'].iloc[0]
                    event_observed = 1
                    censored = 0
                else:
                    event_time = woman_data['孕周_数值'].iloc[-1]
                    event_observed = 0
                    censored = 1
                
                # 计算多因素特征
                max_concentration = woman_data['Y染色体浓度'].max()
                min_concentration = woman_data['Y染色体浓度'].min()
                concentration_trend = self.calculate_concentration_trend(woman_data)
                concentration_variability = woman_data['Y染色体浓度'].std()
                检测次数 = len(woman_data)
                
                # 检测时间跨度
                detection_span = woman_data['孕周_数值'].max() - woman_data['孕周_数值'].min()
                
                survival_data.append({
                    '孕妇代码': woman_code,
                    'BMI': bmi,
                    '年龄': age,
                    '身高': height,
                    '体重': weight,
                    '怀孕次数': pregnancy_count,
                    '生产次数': birth_count,
                    'IVF妊娠': ivf_pregnancy,
                    '事件时间': event_time,
                    '事件观察': event_observed,
                    '右删失': censored,
                    '最大浓度': max_concentration,
                    '最小浓度': min_concentration,
                    '浓度趋势': concentration_trend,
                    '浓度变异性': concentration_variability,
                    '检测次数': 检测次数,
                    '检测时间跨度': detection_span,
                    '第一次检测孕周': woman_data['孕周_数值'].iloc[0],
                    '最后检测孕周': woman_data['孕周_数值'].iloc[-1],
                    '平均GC含量': avg_gc_content,
                    '平均比对率': avg_mapping_rate,
                    '平均重复率': avg_duplicate_rate,
                    '平均唯一读段数': avg_unique_reads,
                    '平均过滤率': avg_filtered_rate,
                    '平均13号染色体Z值': avg_chr13_z,
                    '平均18号染色体Z值': avg_chr18_z,
                    '平均21号染色体Z值': avg_chr21_z,
                    '平均X染色体浓度': avg_x_concentration
                })
            
            self.survival_df = pd.DataFrame(survival_data)
            self.original_male_data = male_data
            
            # 数据统计
            print(f"多因素生存数据构建完成:")
            print(f"总孕妇数: {len(self.survival_df)}")
            print(f"观察到达标事件: {self.survival_df['事件观察'].sum()}")
            print(f"右删失: {self.survival_df['右删失'].sum()}")
            print(f"达标率: {self.survival_df['事件观察'].mean():.1%}")
            print(f"平均检测次数: {self.survival_df['检测次数'].mean():.1f}")
            
            return True
            
        except Exception as e:
            print(f"数据处理失败: {e}")
            traceback.print_exc()
            return False
    
    def calculate_concentration_trend(self, woman_data):
        """计算Y染色体浓度变化趋势"""
        if len(woman_data) < 2:
            return 0
        
        x = woman_data['孕周_数值'].values
        y = woman_data['Y染色体浓度'].values
        
        # 线性回归斜率作为趋势指标
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    def multifactor_correlation_analysis(self):
        """多因素相关性分析"""
        print("\n" + "="*60)
        print("步骤1: 多因素相关性分析")
        print("="*60)
        
        # 构建多因素特征矩阵
        basic_factors = ['BMI', '年龄', '身高', '体重', '怀孕次数', '生产次数', 'IVF妊娠']
        concentration_factors = ['最大浓度', '最小浓度', '浓度趋势', '浓度变异性']
        detection_factors = ['检测次数', '检测时间跨度', '第一次检测孕周']
        technical_factors = ['平均GC含量', '平均比对率', '平均重复率', '平均唯一读段数', '平均过滤率']
        chromosome_factors = ['平均13号染色体Z值', '平均18号染色体Z值', '平均21号染色体Z值', '平均X染色体浓度']
        
        all_factors = basic_factors + concentration_factors + detection_factors + technical_factors + chromosome_factors
        available_factors = [f for f in all_factors if f in self.survival_df.columns and self.survival_df[f].notna().sum() > 10]
        
        # 添加衍生特征
        if 'BMI' in self.survival_df.columns and '年龄' in self.survival_df.columns:
            # 确保数据类型正确
            self.survival_df['BMI'] = pd.to_numeric(self.survival_df['BMI'], errors='coerce')
            self.survival_df['年龄'] = pd.to_numeric(self.survival_df['年龄'], errors='coerce')
            self.survival_df['BMI_年龄交互'] = self.survival_df['BMI'] * self.survival_df['年龄']
            available_factors.append('BMI_年龄交互')
        
        if '身高' in self.survival_df.columns and '体重' in self.survival_df.columns:
            # 确保数据类型正确
            self.survival_df['身高'] = pd.to_numeric(self.survival_df['身高'], errors='coerce')
            self.survival_df['体重'] = pd.to_numeric(self.survival_df['体重'], errors='coerce')
            self.survival_df['身高体重比'] = self.survival_df['身高'] / self.survival_df['体重']
            # 体质指数分类使用数值编码而不是分类
            self.survival_df['体质指数分类'] = pd.cut(self.survival_df['BMI'], 
                                               bins=[0, 18.5, 24, 28, 32, 50], 
                                               labels=[1, 2, 3, 4, 5]).astype(float)
            available_factors.extend(['身高体重比', '体质指数分类'])
        
        if '怀孕次数' in self.survival_df.columns and '生产次数' in self.survival_df.columns:
            # 确保数据类型正确
            self.survival_df['怀孕次数'] = pd.to_numeric(self.survival_df['怀孕次数'], errors='coerce')
            self.survival_df['生产次数'] = pd.to_numeric(self.survival_df['生产次数'], errors='coerce')
            self.survival_df['生育经验'] = self.survival_df['怀孕次数'] - self.survival_df['生产次数']
            available_factors.append('生育经验')
        
        if '检测次数' in self.survival_df.columns and '检测时间跨度' in self.survival_df.columns:
            # 确保数据类型正确
            self.survival_df['检测次数'] = pd.to_numeric(self.survival_df['检测次数'], errors='coerce')
            self.survival_df['检测时间跨度'] = pd.to_numeric(self.survival_df['检测时间跨度'], errors='coerce')
            self.survival_df['检测密度'] = self.survival_df['检测次数'] / (self.survival_df['检测时间跨度'] + 1)
            available_factors.append('检测密度')
        
        # 计算相关系数矩阵
        correlation_data = self.survival_df[available_factors + ['事件时间', '事件观察']].copy()
        
        # 处理缺失值 - 区分数值和分类变量
        for col in correlation_data.columns:
            if correlation_data[col].dtype.name == 'category':
                # 分类变量用众数填充
                mode_val = correlation_data[col].mode()
                if len(mode_val) > 0:
                    correlation_data[col] = correlation_data[col].fillna(mode_val[0])
                else:
                    # 如果没有众数，转换为数值
                    correlation_data[col] = pd.to_numeric(correlation_data[col], errors='coerce')
                    correlation_data[col] = correlation_data[col].fillna(correlation_data[col].median())
            else:
                # 数值变量用中位数填充
                correlation_data[col] = pd.to_numeric(correlation_data[col], errors='coerce')
                correlation_data[col] = correlation_data[col].fillna(correlation_data[col].median())
        
        correlation_matrix = correlation_data.corr()
        
        print(f"分析的因素数量: {len(available_factors)}")
        print(f"因素类别:")
        print(f"  基础生物学因素: {[f for f in basic_factors if f in available_factors]}")
        print(f"  浓度相关因素: {[f for f in concentration_factors if f in available_factors]}")
        print(f"  检测相关因素: {[f for f in detection_factors if f in available_factors]}")
        print(f"  技术质量因素: {[f for f in technical_factors if f in available_factors]}")
        print(f"  染色体异常因素: {[f for f in chromosome_factors if f in available_factors]}")
        
        print(f"\n与事件时间的相关性:")
        time_corr = correlation_matrix['事件时间'].drop('事件时间').sort_values(key=abs, ascending=False)
        for factor, corr in time_corr.items():
            if abs(corr) > 0.05:  # 只显示有一定相关性的因素
                significance = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
                print(f"  {factor}: {corr:.3f} {significance}")
        
        print(f"\n与事件观察的相关性:")
        event_corr = correlation_matrix['事件观察'].drop('事件观察').sort_values(key=abs, ascending=False)
        for factor, corr in event_corr.items():
            if abs(corr) > 0.05:  # 只显示有一定相关性的因素
                significance = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
                print(f"  {factor}: {corr:.3f} {significance}")
        
        # 保存相关性矩阵
        self.correlation_matrix = correlation_matrix
        self.significant_factors = [f for f in available_factors 
                                  if abs(correlation_matrix.loc[f, '事件时间']) > 0.1 
                                  or abs(correlation_matrix.loc[f, '事件观察']) > 0.1]
        
        # 如果显著因素太少，降低阈值
        if len(self.significant_factors) < 5:
            self.significant_factors = [f for f in available_factors 
                                      if abs(correlation_matrix.loc[f, '事件时间']) > 0.05 
                                      or abs(correlation_matrix.loc[f, '事件观察']) > 0.05]
        
        print(f"\n显著影响因素 (|r| > 0.1 或 前10名): {self.significant_factors[:10]}")
        
        # 创建相关性热图
        self.create_correlation_heatmap()
        
        return True
    
    def bmi_based_grouping_with_multifactor_optimization(self):
        """基于BMI的不重叠分组，结合多因素优化"""
        print("\n" + "="*60)
        print("步骤2: 基于BMI的不重叠分组与多因素优化")
        print("="*60)
        
        # 清理BMI数据
        self.survival_df['BMI'] = pd.to_numeric(self.survival_df['BMI'], errors='coerce')
        self.survival_df = self.survival_df.dropna(subset=['BMI'])
        
        print(f"BMI数据范围: {self.survival_df['BMI'].min():.1f} - {self.survival_df['BMI'].max():.1f}")
        
        # 方法1: 基于BMI分位数的分组
        def create_quantile_groups():
            """基于分位数创建BMI分组"""
            bmi_values = self.survival_df['BMI'].sort_values()
            n_samples = len(bmi_values)
            
            # 尝试4-6组的分位数分组
            best_score = -1
            best_groups = None
            best_boundaries = None
            
            for n_groups in range(4, 7):
                # 计算分位数边界
                quantiles = np.linspace(0, 1, n_groups + 1)
                boundaries = [bmi_values.quantile(q) for q in quantiles]
                boundaries[0] = bmi_values.min() - 0.1  # 确保包含最小值
                boundaries[-1] = bmi_values.max() + 0.1  # 确保包含最大值
                
                # 分配组别
                group_labels = pd.cut(self.survival_df['BMI'], bins=boundaries, 
                                    labels=range(n_groups), include_lowest=True)
                
                # 评估分组质量
                group_sizes = group_labels.value_counts().sort_index()
                min_group_size = group_sizes.min()
                
                # 计算组间差异
                group_times = []
                group_rates = []
                for i in range(n_groups):
                    group_data = self.survival_df[group_labels == i]
                    if len(group_data) > 0:
                        observed_data = group_data[group_data['事件观察'] == 1]
                        if len(observed_data) > 0:
                            group_times.append(observed_data['事件时间'].mean())
                        else:
                            group_times.append(15.0)  # 默认值
                        group_rates.append(group_data['事件观察'].mean())
                
                # 评分：考虑组大小均衡性和组间差异
                size_balance = 1 - (group_sizes.std() / group_sizes.mean()) if group_sizes.mean() > 0 else 0
                time_variance = np.var(group_times) if len(group_times) > 1 else 0
                rate_variance = np.var(group_rates) if len(group_rates) > 1 else 0
                
                score = 0.4 * size_balance + 0.3 * min(time_variance/10, 1) + 0.3 * rate_variance
                
                if score > best_score and min_group_size >= 10:  # 至少10个样本每组
                    best_score = score
                    best_groups = group_labels
                    best_boundaries = boundaries
                    
                print(f"  {n_groups}组方案: 最小组大小={min_group_size}, 评分={score:.3f}")
            
            return best_groups, best_boundaries
        
        # 方法2: 临床意义的BMI分组（仅基于BMI）
        def create_clinical_groups():
            """基于临床意义创建BMI分组（仅考虑BMI）"""
            # WHO BMI分类标准
            boundaries = [0, 18.5, 24, 28, 32, 50]  # 偏瘦、正常、超重、肥胖I、肥胖II及以上
            group_names = ['偏瘦', '正常', '超重', '肥胖I度', '肥胖II度及以上']
            
            # 检查每组样本数
            group_labels = pd.cut(self.survival_df['BMI'], bins=boundaries, 
                                labels=range(len(group_names)), include_lowest=True)
            group_sizes = group_labels.value_counts().sort_index()
            
            # 如果某组样本太少，合并相邻组
            if group_sizes.min() < 15:  # 至少15个样本
                # 合并为4组
                boundaries = [0, 23, 28, 32, 50]
                group_names = ['BMI<23', '23≤BMI<28', '28≤BMI<32', 'BMI≥32']
                group_labels = pd.cut(self.survival_df['BMI'], bins=boundaries,
                                    labels=range(len(group_names)), include_lowest=True)
                group_sizes = group_labels.value_counts().sort_index()
                
                if group_sizes.min() < 10:  # 如果还是太少，进一步合并
                    boundaries = [0, 25, 30, 50]
                    group_names = ['BMI<25', '25≤BMI<30', 'BMI≥30']
                    group_labels = pd.cut(self.survival_df['BMI'], bins=boundaries,
                                        labels=range(len(group_names)), include_lowest=True)
            
            return group_labels, boundaries, group_names
        
        # 执行两种分组方法
        quantile_groups, quantile_boundaries = create_quantile_groups()
        clinical_groups, clinical_boundaries, clinical_names = create_clinical_groups()
        
        # 选择最佳分组方法
        if quantile_groups is not None:
            # 评估两种方法
            quantile_sizes = quantile_groups.value_counts()
            clinical_sizes = clinical_groups.value_counts()
            
            # 选择组间差异更大且组大小更均衡的方法
            if quantile_sizes.min() >= 15 and quantile_sizes.std() / quantile_sizes.mean() < 0.5:
                final_groups = quantile_groups
                final_boundaries = quantile_boundaries
                group_type = "分位数分组"
                group_names = [f"BMI组{i}" for i in range(len(quantile_boundaries)-1)]
            else:
                final_groups = clinical_groups
                final_boundaries = clinical_boundaries
                group_type = "临床分组"
                group_names = clinical_names
        else:
            final_groups = clinical_groups
            final_boundaries = clinical_boundaries
            group_type = "临床分组"
            group_names = clinical_names
        
        self.survival_df['BMI分组'] = final_groups
        
        print(f"\n选择{group_type}方法:")
        print(f"BMI分组边界: {[f'{b:.1f}' for b in final_boundaries]}")
        
        # 输出每组的详细信息
        for i, group_name in enumerate(group_names):
            group_data = self.survival_df[final_groups == i]
            if len(group_data) > 0:
                bmi_range = f"[{group_data['BMI'].min():.1f}, {group_data['BMI'].max():.1f}]"
                observed_data = group_data[group_data['事件观察'] == 1]
                reach_rate = len(observed_data) / len(group_data)
                avg_time = observed_data['事件时间'].mean() if len(observed_data) > 0 else np.nan
                
                print(f"  组{i} ({group_name}): {len(group_data)}个样本, BMI{bmi_range}")
                print(f"    达标率: {reach_rate:.1%}, 平均达标时间: {avg_time:.1f}周")
        
        # 多因素特征分析
        self.analyze_multifactor_characteristics()
        
        return True
    
    def analyze_multifactor_characteristics(self):
        """分析每个BMI组的多因素特征（仅基于BMI分组）"""
        print(f"\n多因素特征分析:")
        print("-" * 60)
        
        # 为每个BMI组计算多因素风险评分
        for group_id in sorted(self.survival_df['BMI分组'].dropna().unique()):
            group_data = self.survival_df[self.survival_df['BMI分组'] == group_id]
            if len(group_data) == 0:
                continue
                
            print(f"\nBMI组 {group_id}:")
            print(f"  样本数: {len(group_data)}")
            
            # 基础特征
            if 'BMI' in group_data.columns:
                print(f"  BMI范围: [{group_data['BMI'].min():.1f}, {group_data['BMI'].max():.1f}]")
            if '年龄' in group_data.columns and group_data['年龄'].notna().sum() > 0:
                print(f"  年龄范围: [{group_data['年龄'].min():.0f}, {group_data['年龄'].max():.0f}]岁")
            
            # 达标特征
            observed_data = group_data[group_data['事件观察'] == 1]
            reach_rate = len(observed_data) / len(group_data)
            print(f"  达标率: {reach_rate:.1%}")
            
            if len(observed_data) > 0:
                avg_time = observed_data['事件时间'].mean()
                time_std = observed_data['事件时间'].std()
                print(f"  平均达标时间: {avg_time:.1f} ± {time_std:.1f}周")
            
            # 多因素风险特征（主要基于BMI）
            risk_factors = []
            
            # BMI风险（主要风险因子）
            avg_bmi = group_data['BMI'].mean()
            bmi_risk = 1 + (avg_bmi - 24) * 0.04  # 24为正常BMI上限，系数调高
            risk_factors.append(('BMI风险', bmi_risk))
            
            # 年龄风险（辅助因子）
            if '年龄' in group_data.columns and group_data['年龄'].notna().sum() > 0:
                avg_age = group_data['年龄'].mean()
                age_risk = 1 + (avg_age - 28) * 0.015  # 降低年龄权重
                risk_factors.append(('年龄风险', age_risk))
            
            # 检测复杂性风险（辅助因子）
            if '检测次数' in group_data.columns:
                avg_detections = group_data['检测次数'].mean()
                detection_risk = 1 + (avg_detections - 3) * 0.03  # 降低检测复杂性权重
                risk_factors.append(('检测复杂性', detection_risk))
            
            # 浓度风险（辅助因子）
            if '最小浓度' in group_data.columns:
                avg_min_conc = group_data['最小浓度'].mean()
                conc_risk = 1 + (0.04 - avg_min_conc) * 3  # 降低浓度风险权重
                risk_factors.append(('浓度风险', conc_risk))
            
            # 计算综合风险评分（主要基于BMI）
            if len(risk_factors) > 0:
                # BMI权重70%，其他因素权重30%
                bmi_weight = 0.7
                other_weight = 0.3 / (len(risk_factors) - 1) if len(risk_factors) > 1 else 0
                
                weighted_risk = risk_factors[0][1] * bmi_weight  # BMI风险
                for _, risk in risk_factors[1:]:  # 其他风险
                    weighted_risk += risk * other_weight
                
                total_risk = weighted_risk
            else:
                total_risk = 1.0
                
            print(f"  综合风险评分: {total_risk:.2f}")
            
            # 保存风险评分
            self.survival_df.loc[self.survival_df['BMI分组'] == group_id, '风险评分'] = total_risk
    
    def optimize_kmeans_clustering(self, data):
        """优化K-means聚类"""
        k_range = range(2, min(12, len(data)//8))  # 增加聚类数量范围
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            
            silhouette = silhouette_score(data, labels)
            calinski = calinski_harabasz_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)
            
            silhouette_scores.append(silhouette)
            calinski_scores.append(calinski)
            davies_bouldin_scores.append(davies_bouldin)
            
            print(f"  K={k}: 轮廓系数={silhouette:.3f}, Calinski-Harabasz={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}")
        
        if silhouette_scores:
            # 综合评分 - 平衡轮廓系数和Davies-Bouldin指数
            silhouette_norm = np.array(silhouette_scores) / max(silhouette_scores)
            calinski_norm = np.array(calinski_scores) / max(calinski_scores)
            davies_bouldin_norm = 1 - (np.array(davies_bouldin_scores) / max(davies_bouldin_scores))  # 越小越好，所以取反
            
            # 加权综合评分，偏向更多聚类组（在合理范围内）
            k_preference = np.array([(k-2)*0.1 for k in k_range])  # 偏向更多组
            combined_scores = 0.4 * silhouette_norm + 0.3 * calinski_norm + 0.2 * davies_bouldin_norm + 0.1 * k_preference
            
            best_k = k_range[np.argmax(combined_scores)]
            print(f"  K-means最优k: {best_k}")
            return best_k
        
        return 0
    
    def optimize_hierarchical_clustering(self, data):
        """优化层次聚类"""
        k_range = range(2, min(10, len(data)//12))  # 增加聚类数量范围
        silhouette_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = hierarchical.fit_predict(data)
            
            silhouette = silhouette_score(data, labels)
            davies_bouldin = davies_bouldin_score(data, labels)
            
            silhouette_scores.append(silhouette)
            davies_bouldin_scores.append(davies_bouldin)
            
            print(f"  K={k}: 轮廓系数={silhouette:.3f}, Davies-Bouldin={davies_bouldin:.3f}")
        
        if silhouette_scores:
            # 综合评分
            silhouette_norm = np.array(silhouette_scores) / max(silhouette_scores)
            davies_bouldin_norm = 1 - (np.array(davies_bouldin_scores) / max(davies_bouldin_scores))
            
            # 偏向更多聚类组
            k_preference = np.array([(k-2)*0.15 for k in k_range])
            combined_scores = 0.5 * silhouette_norm + 0.3 * davies_bouldin_norm + 0.2 * k_preference
            
            best_k = k_range[np.argmax(combined_scores)]
            print(f"  层次聚类最优k: {best_k}")
            return best_k
        
        return 0
    
    def optimize_dbscan_clustering(self, data):
        """优化DBSCAN聚类"""
        eps_range = np.arange(0.3, 2.0, 0.2)
        min_samples_range = range(5, min(20, len(data)//10))
        
        best_score = -1
        best_labels = None
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 1 and n_noise < len(data) * 0.5:  # 噪声点不超过50%
                    try:
                        silhouette = silhouette_score(data, labels)
                        if silhouette > best_score:
                            best_score = silhouette
                            best_labels = labels
                    except:
                        continue
        
        if best_labels is not None:
            n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise = list(best_labels).count(-1)
            print(f"  DBSCAN最优: 聚类数={n_clusters}, 噪声点={n_noise}, 轮廓系数={best_score:.3f}")
            
        return best_labels
    
    def select_best_clustering(self, data, clustering_results):
        """选择最佳聚类方法"""
        if not clustering_results:
            return None
        
        method_scores = {}
        
        for method_name, (n_clusters, labels) in clustering_results.items():
            try:
                # 综合评估：轮廓系数 + 组间差异 + 样本分布均匀性
                silhouette = silhouette_score(data, labels)
                
                # 计算组间事件时间差异
                group_times = []
                for cluster_id in set(labels):
                    if cluster_id != -1:  # 排除噪声点
                        cluster_mask = np.array(labels) == cluster_id
                        cluster_event_times = self.survival_df.loc[cluster_mask, '事件时间']
                        if len(cluster_event_times) > 0:
                            group_times.append(cluster_event_times.mean())
                
                time_variance = np.var(group_times) if len(group_times) > 1 else 0
                
                # 样本分布均匀性
                unique_labels, counts = np.unique(labels, return_counts=True)
                valid_counts = counts[unique_labels != -1] if -1 in unique_labels else counts
                distribution_score = 1 - np.std(valid_counts) / np.mean(valid_counts) if len(valid_counts) > 1 else 0
                
                # 综合评分
                combined_score = 0.4 * silhouette + 0.4 * min(time_variance/10, 1) + 0.2 * distribution_score
                method_scores[method_name] = combined_score
                
                print(f"  {method_name}: 轮廓系数={silhouette:.3f}, 时间方差={time_variance:.2f}, 分布评分={distribution_score:.3f}, 综合评分={combined_score:.3f}")
                
            except Exception as e:
                print(f"  {method_name}: 评估失败 - {e}")
                continue
        
        if method_scores:
            best_method_name = max(method_scores, key=method_scores.get)
            return best_method_name, clustering_results[best_method_name]
        
        return None
    
    def analyze_cluster_characteristics(self, features):
        """分析聚类组特征"""
        print(f"\n聚类组特征分析:")
        print("-" * 60)
        
        for cluster_id in sorted(self.survival_df['多因素聚类组'].unique()):
            if cluster_id == -1:  # 跳过噪声点
                continue
                
            cluster_data = self.survival_df[self.survival_df['多因素聚类组'] == cluster_id]
            observed_data = cluster_data[cluster_data['事件观察'] == 1]
            
            print(f"\n聚类组 {cluster_id}:")
            print(f"  样本数: {len(cluster_data)}")
            print(f"  达标数: {len(observed_data)}")
            print(f"  达标率: {len(observed_data)/len(cluster_data):.1%}")
            
            if len(observed_data) > 0:
                print(f"  平均达标时间: {observed_data['事件时间'].mean():.1f}周")
            
            # 显示各因素的均值
            for feature in features:
                if feature in cluster_data.columns:
                    mean_val = cluster_data[feature].mean()
                    print(f"  {feature}: {mean_val:.2f}")
    
    def fallback_to_bmi_grouping(self):
        """改进的多维度分组策略"""
        print("使用改进的多维度分组策略")
        
        # 创建更细致的多维度分组
        def assign_multifactor_group(row):
            bmi = row['BMI']
            age = row['年龄'] if not pd.isna(row['年龄']) else 30
            检测次数 = row['检测次数'] if not pd.isna(row['检测次数']) else 3
            最大浓度 = row['最大浓度'] if not pd.isna(row['最大浓度']) else 0.05
            
            # BMI分组（更细致）
            if pd.isna(bmi):
                bmi_group = 2
            elif bmi < 18.5:
                bmi_group = 0  # 偏瘦
            elif bmi < 24:
                bmi_group = 1  # 正常
            elif bmi < 28:
                bmi_group = 2  # 超重
            elif bmi < 32:
                bmi_group = 3  # 肥胖I度
            elif bmi < 36:
                bmi_group = 4  # 肥胖II度
            else:
                bmi_group = 5  # 肥胖III度
            
            # 年龄分组
            if age < 25:
                age_group = 0  # 年轻
            elif age < 30:
                age_group = 1  # 青年
            elif age < 35:
                age_group = 2  # 中年
            else:
                age_group = 3  # 高龄
            
            # 浓度水平分组
            if 最大浓度 < 0.04:
                conc_group = 0  # 低浓度
            elif 最大浓度 < 0.08:
                conc_group = 1  # 中浓度
            else:
                conc_group = 2  # 高浓度
            
            # 综合分组策略 - 创建6个主要组
            if bmi_group <= 1:  # 正常或偏瘦
                if age_group <= 1:  # 年轻
                    return 0  # 年轻正常体重组
                else:
                    return 1  # 高龄正常体重组
            elif bmi_group <= 3:  # 超重到轻度肥胖
                if age_group <= 1:  # 年轻
                    if conc_group <= 1:
                        return 2  # 年轻超重低浓度组
                    else:
                        return 3  # 年轻超重高浓度组
                else:
                    return 4  # 高龄超重组
            else:  # 重度肥胖
                return 5  # 重度肥胖组
        
        self.survival_df['多因素聚类组'] = self.survival_df.apply(assign_multifactor_group, axis=1)
        
        print(f"分组结果:")
        group_descriptions = {
            0: "年轻正常体重组",
            1: "高龄正常体重组", 
            2: "年轻超重低浓度组",
            3: "年轻超重高浓度组",
            4: "高龄超重组",
            5: "重度肥胖组"
        }
        
        for group_id in sorted(self.survival_df['多因素聚类组'].unique()):
            group_data = self.survival_df[self.survival_df['多因素聚类组'] == group_id]
            desc = group_descriptions.get(group_id, f"组{group_id}")
            print(f"  {desc}: {len(group_data)}个样本")
        
        return True
    
    def machine_learning_risk_prediction(self):
        """机器学习风险预测模型"""
        print("\n" + "="*60)
        print("步骤3: 机器学习风险预测模型")
        print("="*60)
        
        # 准备特征和目标变量
        group_column = 'BMI分组' if 'BMI分组' in self.survival_df.columns else '多因素聚类组'
        feature_cols = self.significant_factors + [group_column]
        
        # 确保所有特征都存在
        available_features = [f for f in feature_cols if f in self.survival_df.columns]
        X = self.survival_df[available_features].copy()
        
        # 处理缺失值 - 分别处理数值型和分类型数据
        for col in X.columns:
            if X[col].dtype.name == 'category':
                # 分类数据用众数填充
                mode_val = X[col].mode()
                if len(mode_val) > 0:
                    X[col] = X[col].fillna(mode_val[0])
                # 转换为数值编码
                X[col] = X[col].cat.codes
            else:
                # 数值数据用均值填充
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].mean())
        
        # 目标变量1：达标时间预测（回归）
        y_time = self.survival_df['事件时间'].copy()
        
        # 目标变量2：达标概率预测（分类）
        y_reach = self.survival_df['事件观察'].copy()
        
        print(f"特征数量: {len(available_features)}")
        print(f"样本数量: {len(X)}")
        
        # 构建多种模型
        models_regression = {
            '随机森林回归': RandomForestRegressor(n_estimators=100, random_state=42),
            '梯度提升回归': GradientBoostingRegressor(random_state=42),
            '支持向量回归': SVR(kernel='rbf'),
            '神经网络回归': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        models_classification = {
            '随机森林分类': RandomForestClassifier(n_estimators=100, random_state=42),
            '逻辑回归': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # 训练达标时间预测模型
        print(f"\n训练达标时间预测模型:")
        self.time_prediction_models = {}
        
        for name, model in models_regression.items():
            try:
                # 交叉验证评估
                cv_scores = cross_val_score(model, X, y_time, cv=5, scoring='neg_mean_squared_error')
                mse = -cv_scores.mean()
                rmse = np.sqrt(mse)
                
                # 训练最终模型
                model.fit(X, y_time)
                self.time_prediction_models[name] = model
                
                print(f"  {name}: RMSE={rmse:.2f}")
                
                # 特征重要性（如果支持）
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    top_features = sorted(zip(available_features, importance), key=lambda x: x[1], reverse=True)[:3]
                    print(f"    重要特征: {[f'{feat}({imp:.3f})' for feat, imp in top_features]}")
                    
            except Exception as e:
                print(f"  {name}: 训练失败 - {e}")
        
        # 训练达标概率预测模型
        print(f"\n训练达标概率预测模型:")
        self.reach_prediction_models = {}
        
        for name, model in models_classification.items():
            try:
                # 交叉验证评估
                cv_scores = cross_val_score(model, X, y_reach, cv=5, scoring='accuracy')
                accuracy = cv_scores.mean()
                
                # 训练最终模型
                model.fit(X, y_reach)
                self.reach_prediction_models[name] = model
                
                print(f"  {name}: 准确率={accuracy:.3f}")
                
                # 特征重要性（如果支持）
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    top_features = sorted(zip(available_features, importance), key=lambda x: x[1], reverse=True)[:3]
                    print(f"    重要特征: {[f'{feat}({imp:.3f})' for feat, imp in top_features]}")
                    
            except Exception as e:
                print(f"  {name}: 训练失败 - {e}")
        
        return True
    
    def monte_carlo_error_simulation(self, n_simulations=1000):
        """蒙特卡洛检测误差仿真"""
        print("\n" + "="*60)
        print("步骤4: 蒙特卡洛检测误差仿真")
        print("="*60)
        
        # 为每个BMI组进行误差仿真
        self.error_simulation_results = {}
        group_column = 'BMI分组' if 'BMI分组' in self.survival_df.columns else '多因素聚类组'
        
        for group_id in sorted(self.survival_df[group_column].dropna().unique()):
            if group_id == -1:
                continue
                
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            if len(group_data) == 0:
                continue
            
            print(f"\nBMI组 {group_id} 误差仿真:")
            
            # 基础参数
            base_reach_rate = group_data['事件观察'].mean()
            base_mean_time = group_data[group_data['事件观察'] == 1]['事件时间'].mean()
            
            if pd.isna(base_mean_time):
                base_mean_time = 15.0
            
            # 蒙特卡洗仿真
            simulation_results = []
            
            for _ in range(n_simulations):
                # 添加测量误差
                true_concentrations = np.random.normal(self.threshold, self.measurement_error_std * 2, len(group_data))
                measurement_errors = np.random.normal(0, self.measurement_error_std, len(group_data))
                measured_concentrations = true_concentrations + measurement_errors
                
                # 计算在误差影响下的达标情况
                simulated_reach = (measured_concentrations >= self.threshold).astype(int)
                simulated_reach_rate = simulated_reach.mean()
                
                # 计算时间偏移
                time_error = np.random.normal(0, 0.5, 1)[0]  # 时间测量误差
                simulated_mean_time = base_mean_time + time_error
                
                simulation_results.append({
                    'reach_rate': simulated_reach_rate,
                    'mean_time': simulated_mean_time,
                    'false_positive_rate': max(0, simulated_reach_rate - base_reach_rate),
                    'false_negative_rate': max(0, base_reach_rate - simulated_reach_rate)
                })
            
            # 统计仿真结果
            sim_df = pd.DataFrame(simulation_results)
            
            error_stats = {
                'base_reach_rate': base_reach_rate,
                'base_mean_time': base_mean_time,
                'reach_rate_mean': sim_df['reach_rate'].mean(),
                'reach_rate_std': sim_df['reach_rate'].std(),
                'reach_rate_ci_lower': sim_df['reach_rate'].quantile(0.05),
                'reach_rate_ci_upper': sim_df['reach_rate'].quantile(0.95),
                'time_mean': sim_df['mean_time'].mean(),
                'time_std': sim_df['mean_time'].std(),
                'time_ci_lower': sim_df['mean_time'].quantile(0.05),
                'time_ci_upper': sim_df['mean_time'].quantile(0.95),
                'false_positive_rate': sim_df['false_positive_rate'].mean(),
                'false_negative_rate': sim_df['false_negative_rate'].mean()
            }
            
            self.error_simulation_results[group_id] = error_stats
            
            print(f"  达标率: {base_reach_rate:.3f} → {error_stats['reach_rate_mean']:.3f} ± {error_stats['reach_rate_std']:.3f}")
            print(f"  达标时间: {base_mean_time:.1f} → {error_stats['time_mean']:.1f} ± {error_stats['time_std']:.1f}")
            print(f"  假阳性率: {error_stats['false_positive_rate']:.3f}")
            print(f"  假阴性率: {error_stats['false_negative_rate']:.3f}")
        
        return True
    
    def comprehensive_risk_optimization(self):
        """综合风险优化"""
        print("\n" + "="*60)
        print("步骤5: 综合风险优化与最佳时点计算")
        print("="*60)
        
        recommendations = []
        
        # 使用BMI分组而不是聚类组
        group_column = 'BMI分组' if 'BMI分组' in self.survival_df.columns else '多因素聚类组'
        
        for group_id in sorted(self.survival_df[group_column].dropna().unique()):
            if group_id == -1:
                continue
                
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            observed_data = group_data[group_data['事件观察'] == 1]
            
            if len(group_data) == 0:
                continue
            
            print(f"\nBMI组 {group_id} 风险优化:")
            
            # 基础统计
            sample_size = len(group_data)
            reach_count = len(observed_data)
            reach_rate = reach_count / sample_size
            
            # 组特征描述 - 使用BMI范围
            bmi_min = group_data['BMI'].min()
            bmi_max = group_data['BMI'].max()
            group_description = f"BMI[{bmi_min:.1f}-{bmi_max:.1f}]"
            
            if '年龄' in group_data.columns and group_data['年龄'].notna().sum() > 0:
                age_min = group_data['年龄'].min()
                age_max = group_data['年龄'].max()
                group_description += f", 年龄[{age_min:.0f}-{age_max:.0f}]"
            
            # 计算最佳时点
            if len(observed_data) >= 3:
                # 多种时点计算方法
                mean_time = observed_data['事件时间'].mean()
                median_time = observed_data['事件时间'].median()
                percentile_80 = observed_data['事件时间'].quantile(0.8)
                percentile_90 = observed_data['事件时间'].quantile(0.9)
                
                # 考虑检测误差的调整
                error_adjustment = 0
                if group_id in self.error_simulation_results:
                    error_stats = self.error_simulation_results[group_id]
                    # 基于假阴性率调整（假阴性率高则提前检测）
                    error_adjustment = error_stats['false_negative_rate'] * 1.5
                
                # 综合风险函数优化
                def comprehensive_risk_function(t):
                    # 早期检测风险 (t < 12周风险较低)
                    early_risk = max(0, 12 - t) * 0.1
                    
                    # 晚期检测风险 (t > 18周风险较高)
                    late_risk = max(0, t - 18) * 0.3
                    
                    # 未达标风险（基于达标率和时间分布）
                    miss_prob = 1 - min(reach_rate * (t - 10) / 8, 1)  # 简化模型
                    miss_risk = miss_prob * 2.0
                    
                    # 检测误差风险
                    if group_id in self.error_simulation_results:
                        error_stats = self.error_simulation_results[group_id]
                        error_risk = (error_stats['false_negative_rate'] + error_stats['false_positive_rate']) * 0.5
                    else:
                        error_risk = 0.1
                    
                    return early_risk + late_risk + miss_risk + error_risk
                
                # 优化最佳时点
                result = minimize_scalar(comprehensive_risk_function, bounds=(11, 22), method='bounded')
                risk_optimal_time = result.x
                
                # 最终推荐时点（综合多种方法）
                candidate_times = [mean_time, percentile_80, risk_optimal_time]
                # 去除异常值
                candidate_times = [t for t in candidate_times if 11 <= t <= 22]
                
                if candidate_times:
                    final_recommendation = np.median(candidate_times)
                else:
                    final_recommendation = 15.0
                
                # 确保在合理范围内
                final_recommendation = max(12.0, min(20.0, final_recommendation))
                
            else:
                # 样本不足时的默认策略
                final_recommendation = 15.0
                mean_time = np.nan
                percentile_80 = np.nan
                risk_optimal_time = np.nan
            
            # 计算风险因子
            base_risk = 1.0
            
            # 基于达标率的风险调整
            reach_risk_factor = 2 - reach_rate  # 达标率低则风险高
            
            # 基于检测误差的风险调整
            error_risk_factor = 1.0
            if group_id in self.error_simulation_results:
                error_stats = self.error_simulation_results[group_id]
                error_risk_factor = 1 + (error_stats['false_negative_rate'] + error_stats['false_positive_rate'])
            
            # 基于时间变异的风险调整
            time_variance_factor = 1.0
            if len(observed_data) > 1:
                time_cv = observed_data['事件时间'].std() / observed_data['事件时间'].mean()
                time_variance_factor = 1 + time_cv * 0.5
            
            # 综合风险因子
            comprehensive_risk_factor = base_risk * reach_risk_factor * error_risk_factor * time_variance_factor
            comprehensive_risk_factor = max(0.5, min(3.0, comprehensive_risk_factor))  # 限制范围
            
            recommendation = {
                'BMI组ID': group_id,
                '组描述': group_description,
                '样本数量': sample_size,
                '达标数量': reach_count,
                '达标率': f"{reach_rate:.1%}",
                '平均达标时间': f"{mean_time:.1f}周" if not pd.isna(mean_time) else "N/A",
                '80%分位数时点': f"{percentile_80:.1f}周" if not pd.isna(percentile_80) else "N/A",
                '风险优化时点': f"{risk_optimal_time:.1f}周" if not pd.isna(risk_optimal_time) else "N/A",
                '综合风险因子': f"{comprehensive_risk_factor:.2f}",
                '推荐检测时点': f"{final_recommendation:.1f}周",
                '检测误差影响': self.get_error_impact_description(group_id),
                '优化方法': 'BMI分组+多因素优化'
            }
            
            recommendations.append(recommendation)
            
            print(f"  {group_description}")
            print(f"  样本数量: {sample_size}, 达标率: {reach_rate:.1%}")
            print(f"  综合风险因子: {comprehensive_risk_factor:.2f}")
            print(f"  🎯 推荐检测时点: {final_recommendation:.1f}周")
        
        self.multifactor_recommendations = pd.DataFrame(recommendations)
        return self.multifactor_recommendations
    
    def get_error_impact_description(self, cluster_id):
        """获取检测误差影响描述"""
        if cluster_id not in self.error_simulation_results:
            return "误差影响未评估"
        
        error_stats = self.error_simulation_results[cluster_id]
        fp_rate = error_stats['false_positive_rate']
        fn_rate = error_stats['false_negative_rate']
        
        if fp_rate < 0.05 and fn_rate < 0.05:
            return "误差影响较小"
        elif fp_rate > 0.1 or fn_rate > 0.1:
            return "误差影响较大，需要谨慎"
        else:
            return "误差影响中等"
    
    def create_correlation_heatmap(self):
        """创建相关性热图"""
        plt.figure(figsize=(16, 12))
        
        # 重新设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 选择主要因素的相关性
        main_factors = self.significant_factors[:12] + ['事件时间', '事件观察']
        if hasattr(self, 'correlation_matrix'):
            subset_matrix = self.correlation_matrix.loc[main_factors, main_factors]
            
            # 创建更美观的热图
            mask = np.triu(np.ones_like(subset_matrix, dtype=bool))
            
            # 设置图形样式
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # 绘制热图
            sns.heatmap(subset_matrix, annot=True, cmap='RdBu_r', center=0,
                       fmt='.3f', square=True, mask=mask,
                       cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                       linewidths=0.5, ax=ax)
            
            # 美化设置 - 使用英文标题避免字体问题
            ax.set_title('Multifactor Correlation Analysis Heatmap', fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel('Influence Factors', fontsize=14, fontweight='bold')
            ax.set_ylabel('Influence Factors', fontsize=14, fontweight='bold')
            
            # 旋转标签以便阅读
            plt.xticks(rotation=45, ha='right', fontsize=11)
            plt.yticks(rotation=0, fontsize=11)
            
            # 添加网格线
            ax.grid(False)
            
        plt.tight_layout()
        plt.savefig('picture/multifactor_correlation_heatmap.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
        
    def create_clustering_visualization(self):
        """创建BMI分组结果可视化 - 拆分为多个图表"""
        self.create_bmi_distribution_charts()
        self.create_bmi_performance_charts()
        self.create_bmi_risk_charts()
    
    def create_bmi_distribution_charts(self):
        """创建BMI分组分布图表"""
        fig = plt.figure(figsize=(20, 12))
        
        # 使用BMI分组而不是聚类组
        group_column = 'BMI分组' if 'BMI分组' in self.survival_df.columns else '多因素聚类组'
        unique_groups = sorted(self.survival_df[group_column].dropna().unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
        
        # 1. BMI分组分布散点图
        ax1 = fig.add_subplot(2, 3, 1)
        if 'BMI' in self.survival_df.columns and '年龄' in self.survival_df.columns:
            for i, group_id in enumerate(unique_groups):
                if group_id == -1:
                    continue
                group_data = self.survival_df[self.survival_df[group_column] == group_id]
                ax1.scatter(group_data['BMI'], group_data['年龄'], 
                           c=[colors[i]], label=f'BMI组{group_id}', alpha=0.7, s=60, edgecolors='black')
            
            ax1.set_xlabel('BMI指数', fontweight='bold', fontsize=12)
            ax1.set_ylabel('年龄（岁）', fontweight='bold', fontsize=12)
            ax1.set_title('基于BMI的明确分组结果', fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. BMI区间可视化（无重叠）
        ax2 = fig.add_subplot(2, 3, 2)
        group_ranges = []
        group_labels = []
        
        for group_id in unique_groups:
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            if len(group_data) > 0:
                bmi_min = group_data['BMI'].min()
                bmi_max = group_data['BMI'].max()
                group_ranges.append((bmi_min, bmi_max))
                group_labels.append(f'组{group_id}')
        
        # 绘制BMI区间条形图
        y_pos = np.arange(len(group_labels))
        for i, (bmi_min, bmi_max) in enumerate(group_ranges):
            ax2.barh(y_pos[i], bmi_max - bmi_min, left=bmi_min, 
                    color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
            ax2.text(bmi_min + (bmi_max - bmi_min)/2, y_pos[i], 
                    f'[{bmi_min:.1f}, {bmi_max:.1f}]', 
                    ha='center', va='center', fontweight='bold', fontsize=10)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(group_labels)
        ax2.set_xlabel('BMI范围', fontweight='bold', fontsize=12)
        ax2.set_title('BMI分组区间（无重叠）', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. 各组样本分布
        ax3 = fig.add_subplot(2, 3, 3)
        group_counts = self.survival_df[group_column].value_counts().sort_index()
        bars = ax3.bar(range(len(group_counts)), group_counts.values, 
                      color=colors[:len(group_counts)], alpha=0.8, edgecolor='black', linewidth=1)
        
        ax3.set_xlabel('BMI组编号', fontweight='bold', fontsize=12)
        ax3.set_ylabel('样本数量', fontweight='bold', fontsize=12)
        ax3.set_title('各BMI组样本分布', fontweight='bold', fontsize=14)
        ax3.set_xticks(range(len(group_counts)))
        ax3.set_xticklabels([f'组{idx}' for idx in group_counts.index])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, group_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. BMI分布直方图
        ax4 = fig.add_subplot(2, 3, 4)
        for i, group_id in enumerate(unique_groups):
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            ax4.hist(group_data['BMI'], alpha=0.6, label=f'BMI组{group_id}', 
                    color=colors[i], bins=15, edgecolor='black', linewidth=0.5)
        
        ax4.set_xlabel('BMI值', fontweight='bold', fontsize=12)
        ax4.set_ylabel('频数', fontweight='bold', fontsize=12)
        ax4.set_title('各BMI组BMI分布直方图', fontweight='bold', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 年龄分布箱线图
        ax5 = fig.add_subplot(2, 3, 5)
        if '年龄' in self.survival_df.columns:
            age_data = []
            age_labels = []
            for group_id in unique_groups:
                if group_id == -1:
                    continue
                group_data = self.survival_df[self.survival_df[group_column] == group_id]
                if len(group_data) > 0 and group_data['年龄'].notna().sum() > 0:
                    age_data.append(group_data['年龄'].dropna())
                    age_labels.append(f'组{group_id}')
            
            if age_data:
                bp = ax5.boxplot(age_data, labels=age_labels, patch_artist=True)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[i])
                    patch.set_alpha(0.7)
                
                ax5.set_xlabel('BMI组', fontweight='bold', fontsize=12)
                ax5.set_ylabel('年龄（岁）', fontweight='bold', fontsize=12)
                ax5.set_title('各BMI组年龄分布', fontweight='bold', fontsize=14)
                ax5.grid(True, alpha=0.3)
        
        # 6. BMI vs 检测次数散点图
        ax6 = fig.add_subplot(2, 3, 6)
        if '检测次数' in self.survival_df.columns:
            for i, group_id in enumerate(unique_groups):
                if group_id == -1:
                    continue
                group_data = self.survival_df[self.survival_df[group_column] == group_id]
                ax6.scatter(group_data['BMI'], group_data['检测次数'], 
                           c=[colors[i]], label=f'BMI组{group_id}', alpha=0.7, s=50)
            
            ax6.set_xlabel('BMI值', fontweight='bold', fontsize=12)
            ax6.set_ylabel('检测次数', fontweight='bold', fontsize=12)
            ax6.set_title('BMI与检测次数关系', fontweight='bold', fontsize=14)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('picture/bmi_distribution_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()  # 关闭图表避免内存泄漏
    
    def create_bmi_performance_charts(self):
        """创建BMI组性能对比图表"""
        fig = plt.figure(figsize=(16, 12))
        
        group_column = 'BMI分组' if 'BMI分组' in self.survival_df.columns else '多因素聚类组'
        unique_groups = sorted(self.survival_df[group_column].dropna().unique())
        
        # 1. 各组达标率对比
        ax1 = fig.add_subplot(2, 2, 1)
        group_reach_rates = []
        group_labels_reach = []
        
        for group_id in unique_groups:
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            reach_rate = group_data['事件观察'].mean()
            group_reach_rates.append(reach_rate)
            group_labels_reach.append(f'组{group_id}')
        
        bars = ax1.bar(group_labels_reach, group_reach_rates, 
                      color=plt.cm.RdYlGn([r for r in group_reach_rates]), 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax1.set_xlabel('BMI组', fontweight='bold', fontsize=12)
        ax1.set_ylabel('达标率', fontweight='bold', fontsize=12)
        ax1.set_title('各BMI组达标率对比', fontweight='bold', fontsize=14)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加百分比标签
        for bar, rate in zip(bars, group_reach_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 平均达标时间对比
        ax2 = fig.add_subplot(2, 2, 2)
        group_mean_times = []
        
        for group_id in unique_groups:
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            observed_data = group_data[group_data['事件观察'] == 1]
            if len(observed_data) > 0:
                mean_time = observed_data['事件时间'].mean()
                group_mean_times.append(mean_time)
            else:
                group_mean_times.append(0)
        
        bars = ax2.bar(group_labels_reach, group_mean_times,
                      color=plt.cm.viridis_r(np.linspace(0.2, 0.8, len(group_mean_times))),
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('BMI组', fontweight='bold', fontsize=12)
        ax2.set_ylabel('平均达标时间（周）', fontweight='bold', fontsize=12)
        ax2.set_title('各BMI组平均达标时间对比', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, time in zip(bars, group_mean_times):
            if time > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{time:.1f}周', ha='center', va='bottom', fontweight='bold')
        
        # 3. 达标时间分布箱线图
        ax3 = fig.add_subplot(2, 2, 3)
        time_data = []
        time_labels = []
        
        for group_id in unique_groups:
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            observed_data = group_data[group_data['事件观察'] == 1]
            if len(observed_data) > 0:
                time_data.append(observed_data['事件时间'])
                time_labels.append(f'组{group_id}')
        
        if time_data:
            bp = ax3.boxplot(time_data, labels=time_labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(time_data)))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)
            
            ax3.set_xlabel('BMI组', fontweight='bold', fontsize=12)
            ax3.set_ylabel('达标时间（周）', fontweight='bold', fontsize=12)
            ax3.set_title('各BMI组达标时间分布', fontweight='bold', fontsize=14)
            ax3.grid(True, alpha=0.3)
        
        # 4. 达标率vs平均BMI散点图
        ax4 = fig.add_subplot(2, 2, 4)
        avg_bmis = []
        for group_id in unique_groups:
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            avg_bmi = group_data['BMI'].mean()
            avg_bmis.append(avg_bmi)
        
        scatter = ax4.scatter(avg_bmis, group_reach_rates, 
                             c=group_mean_times, cmap='viridis', s=200, alpha=0.7, edgecolors='black')
        
        # 添加组标签
        for i, (bmi, rate) in enumerate(zip(avg_bmis, group_reach_rates)):
            ax4.annotate(f'组{unique_groups[i]}', (bmi, rate), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax4.set_xlabel('平均BMI', fontweight='bold', fontsize=12)
        ax4.set_ylabel('达标率', fontweight='bold', fontsize=12)
        ax4.set_title('BMI vs 达标率关系', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('平均达标时间（周）', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('picture/bmi_performance_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()  # 关闭图表避免内存泄漏
    
    def create_bmi_risk_charts(self):
        """创建BMI风险评估图表"""
        fig = plt.figure(figsize=(16, 10))
        
        group_column = 'BMI分组' if 'BMI分组' in self.survival_df.columns else '多因素聚类组'
        unique_groups = sorted(self.survival_df[group_column].dropna().unique())
        
        # 1. 风险评分对比
        ax1 = fig.add_subplot(2, 3, 1)
        if '风险评分' in self.survival_df.columns:
            group_risk_scores = []
            group_labels = []
            for group_id in unique_groups:
                if group_id == -1:
                    continue
                group_data = self.survival_df[self.survival_df[group_column] == group_id]
                if len(group_data) > 0:
                    risk_score = group_data['风险评分'].mean()
                    group_risk_scores.append(risk_score)
                    group_labels.append(f'组{group_id}')
            
            bars = ax1.bar(group_labels, group_risk_scores,
                          color=plt.cm.Reds([score/max(group_risk_scores) if max(group_risk_scores) > 0 else 0.5 for score in group_risk_scores]),
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            ax1.set_xlabel('BMI组', fontweight='bold', fontsize=12)
            ax1.set_ylabel('综合风险评分', fontweight='bold', fontsize=12)
            ax1.set_title('各BMI组风险评分对比', fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, score in zip(bars, group_risk_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. BMI vs 风险评分
        ax2 = fig.add_subplot(2, 3, 2)
        if '风险评分' in self.survival_df.columns:
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
            for i, group_id in enumerate(unique_groups):
                if group_id == -1:
                    continue
                group_data = self.survival_df[self.survival_df[group_column] == group_id]
                ax2.scatter(group_data['BMI'], group_data['风险评分'], 
                           c=[colors[i]], label=f'BMI组{group_id}', alpha=0.7, s=50)
            
            ax2.set_xlabel('BMI值', fontweight='bold', fontsize=12)
            ax2.set_ylabel('风险评分', fontweight='bold', fontsize=12)
            ax2.set_title('BMI与风险评分关系', fontweight='bold', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 推荐检测时点（如果有的话）
        ax3 = fig.add_subplot(2, 3, 3)
        if hasattr(self, 'multifactor_recommendations'):
            timepoints = [float(x.split('周')[0]) for x in self.multifactor_recommendations['推荐检测时点']]
            groups = self.multifactor_recommendations['BMI组ID']
            
            bars = ax3.bar(range(len(groups)), timepoints, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(timepoints))),
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            ax3.set_xlabel('BMI组', fontweight='bold', fontsize=12)
            ax3.set_ylabel('推荐检测时点（周）', fontweight='bold', fontsize=12)
            ax3.set_title('各BMI组推荐检测时点', fontweight='bold', fontsize=14)
            ax3.set_xticks(range(len(groups)))
            ax3.set_xticklabels([f'组{g}' for g in groups])
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, time in zip(bars, timepoints):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('picture/bmi_risk_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()  # 关闭图表避免内存泄漏
        
        # 使用BMI分组而不是聚类组
        group_column = 'BMI分组' if 'BMI分组' in self.survival_df.columns else '多因素聚类组'
        unique_groups = sorted(self.survival_df[group_column].dropna().unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
        
        # 1. BMI分组分布散点图
        ax1 = fig.add_subplot(2, 3, 1)
        if 'BMI' in self.survival_df.columns and '年龄' in self.survival_df.columns:
            for i, group_id in enumerate(unique_groups):
                if group_id == -1:
                    continue
                group_data = self.survival_df[self.survival_df[group_column] == group_id]
                ax1.scatter(group_data['BMI'], group_data['年龄'], 
                           c=[colors[i]], label=f'BMI Group {group_id}', alpha=0.7, s=60, edgecolors='black')
            
            ax1.set_xlabel('BMI Index', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Age (years)', fontweight='bold', fontsize=12)
            ax1.set_title('BMI-based Clear Grouping Results', fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. BMI区间可视化（无重叠）
        ax2 = fig.add_subplot(2, 3, 2)
        group_ranges = []
        group_labels = []
        
        for group_id in unique_groups:
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            if len(group_data) > 0:
                bmi_min = group_data['BMI'].min()
                bmi_max = group_data['BMI'].max()
                group_ranges.append((bmi_min, bmi_max))
                group_labels.append(f'Group {group_id}')
        
        # 绘制BMI区间条形图
        y_pos = np.arange(len(group_labels))
        for i, (bmi_min, bmi_max) in enumerate(group_ranges):
            ax2.barh(y_pos[i], bmi_max - bmi_min, left=bmi_min, 
                    color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
            ax2.text(bmi_min + (bmi_max - bmi_min)/2, y_pos[i], 
                    f'[{bmi_min:.1f}, {bmi_max:.1f}]', 
                    ha='center', va='center', fontweight='bold', fontsize=10)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(group_labels)
        ax2.set_xlabel('BMI Range', fontweight='bold', fontsize=12)
        ax2.set_title('BMI Group Intervals (Non-overlapping)', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. 各组样本分布
        ax3 = fig.add_subplot(2, 3, 3)
        group_counts = self.survival_df[group_column].value_counts().sort_index()
        bars = ax3.bar(range(len(group_counts)), group_counts.values, 
                      color=colors[:len(group_counts)], alpha=0.8, edgecolor='black', linewidth=1)
        
        ax3.set_xlabel('BMI Group ID', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Sample Count', fontweight='bold', fontsize=12)
        ax3.set_title('Sample Distribution by BMI Groups', fontweight='bold', fontsize=14)
        ax3.set_xticks(range(len(group_counts)))
        ax3.set_xticklabels([f'Group {idx}' for idx in group_counts.index])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, group_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. 各组达标率对比
        ax4 = fig.add_subplot(2, 3, 4)
        group_reach_rates = []
        group_labels_reach = []
        
        for group_id in sorted(unique_groups):
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            reach_rate = group_data['事件观察'].mean()
            group_reach_rates.append(reach_rate)
            group_labels_reach.append(f'Group {group_id}')
        
        bars = ax4.bar(group_labels_reach, group_reach_rates, 
                      color=plt.cm.RdYlGn([r for r in group_reach_rates]), 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax4.set_xlabel('BMI Group', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Achievement Rate', fontweight='bold', fontsize=12)
        ax4.set_title('Achievement Rate Comparison by BMI Groups', fontweight='bold', fontsize=14)
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加百分比标签
        for bar, rate in zip(bars, group_reach_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 5. 平均达标时间对比
        ax5 = fig.add_subplot(2, 3, 5)
        group_mean_times = []
        
        for group_id in sorted(unique_groups):
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            observed_data = group_data[group_data['事件观察'] == 1]
            if len(observed_data) > 0:
                mean_time = observed_data['事件时间'].mean()
                group_mean_times.append(mean_time)
            else:
                group_mean_times.append(0)
        
        bars = ax5.bar(group_labels_reach, group_mean_times,
                      color=plt.cm.viridis_r(np.linspace(0.2, 0.8, len(group_mean_times))),
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax5.set_xlabel('BMI Group', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Average Achievement Time (weeks)', fontweight='bold', fontsize=12)
        ax5.set_title('Average Achievement Time by BMI Groups', fontweight='bold', fontsize=14)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, time in zip(bars, group_mean_times):
            if time > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{time:.1f}w', ha='center', va='bottom', fontweight='bold')
        
        # 6. 风险评分分析
        ax6 = fig.add_subplot(2, 3, 6)
        if '风险评分' in self.survival_df.columns:
            group_risk_scores = []
            for group_id in sorted(unique_groups):
                if group_id == -1:
                    continue
                group_data = self.survival_df[self.survival_df[group_column] == group_id]
                if len(group_data) > 0 and '风险评分' in group_data.columns:
                    risk_score = group_data['风险评分'].mean()
                    group_risk_scores.append(risk_score)
                else:
                    group_risk_scores.append(1.0)
            
            bars = ax6.bar(group_labels_reach, group_risk_scores,
                          color=plt.cm.Reds([score/max(group_risk_scores) if max(group_risk_scores) > 0 else 0.5 for score in group_risk_scores]),
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            ax6.set_xlabel('BMI Group', fontweight='bold', fontsize=12)
            ax6.set_ylabel('Comprehensive Risk Score', fontweight='bold', fontsize=12)
            ax6.set_title('Risk Score Comparison by BMI Groups', fontweight='bold', fontsize=14)
            ax6.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, score in zip(bars, group_risk_scores):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'Risk Score\nCalculating...', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig('picture/bmi_grouping_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
    
    def create_risk_assessment_visualization(self):
        """创建风险评估可视化"""
        if not hasattr(self, 'multifactor_recommendations'):
            return
            
        # 确保中文字体设置生效
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
            
        fig = plt.figure(figsize=(20, 16))
        
        # 获取数据 - 使用BMI组ID
        groups = self.multifactor_recommendations['BMI组ID']
        timepoints = [float(x.split('周')[0]) for x in self.multifactor_recommendations['推荐检测时点']]
        risk_factors = [float(x) for x in self.multifactor_recommendations['综合风险因子']]
        reach_rates = [float(x.strip('%'))/100 for x in self.multifactor_recommendations['达标率']]
        
        # 1. 推荐时点对比 - 美化版
        ax1 = fig.add_subplot(2, 3, 1)
        bars = ax1.bar(groups, timepoints, 
                      color=plt.cm.RdYlGn_r([rf/max(risk_factors) if max(risk_factors) > 0 else 0.5 for rf in risk_factors]), 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax1.set_xlabel('BMI组编号', fontweight='bold')
        ax1.set_ylabel('推荐检测时点（周）', fontweight='bold')
        ax1.set_title('各BMI组最优检测时点推荐', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加详细标签
        for bar, time, risk in zip(bars, timepoints, risk_factors):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{time:.1f}周\n风险系数\n{risk:.2f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. 综合风险因子雷达图
        ax2 = fig.add_subplot(2, 3, 2, projection='polar')
        if len(groups) > 1:
            angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # 闭合
            risk_factors_radar = risk_factors + [risk_factors[0]]  # 闭合
            
            ax2.plot(angles, risk_factors_radar, 'o-', linewidth=3, alpha=0.8, 
                    color='red', markersize=8)
            ax2.fill(angles, risk_factors_radar, alpha=0.25, color='red')
            ax2.set_ylim(0, max(risk_factors) * 1.3 if risk_factors else 1)
            ax2.set_title('综合风险因子分布雷达图', fontweight='bold', fontsize=14, pad=20)
            
            group_labels = [f'BMI组{gid}' for gid in groups]
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(group_labels, fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 添加风险等级标识
            max_risk = max(risk_factors) if risk_factors else 1
            ax2.set_ylim(0, max_risk * 1.2)
            risk_levels = [max_risk * 0.3, max_risk * 0.6, max_risk * 0.9]
            ax2.set_yticks(risk_levels)
            ax2.set_yticklabels(['低风险', '中风险', '高风险'])
        
        # 3. 检测误差影响分析
        ax3 = fig.add_subplot(2, 3, 3)
        if hasattr(self, 'error_simulation_results'):
            error_groups = list(self.error_simulation_results.keys())
            false_positive_rates = [self.error_simulation_results[g]['false_positive_rate'] for g in error_groups]
            false_negative_rates = [self.error_simulation_results[g]['false_negative_rate'] for g in error_groups]
            
            x = np.arange(len(error_groups))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, false_positive_rates, width, 
                           label='假阳性率', alpha=0.8, color='orange', edgecolor='black')
            bars2 = ax3.bar(x + width/2, false_negative_rates, width, 
                           label='假阴性率', alpha=0.8, color='red', edgecolor='black')
            
            ax3.set_xlabel('BMI组', fontweight='bold')
            ax3.set_ylabel('误差率', fontweight='bold')
            ax3.set_title('检测误差影响评估', fontweight='bold', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'组{g}' for g in error_groups])
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. 达标时间置信区间
        ax4 = fig.add_subplot(2, 3, 4)
        if hasattr(self, 'error_simulation_results'):
            groups_ci = []
            means = []
            cis_lower = []
            cis_upper = []
            
            for group_id in sorted(self.error_simulation_results.keys()):
                error_stats = self.error_simulation_results[group_id]
                groups_ci.append(f'组{group_id}')
                means.append(error_stats['time_mean'])
                cis_lower.append(error_stats['time_ci_lower'])
                cis_upper.append(error_stats['time_ci_upper'])
            
            x_pos = np.arange(len(groups_ci))
            
            # 绘制置信区间
            ax4.errorbar(x_pos, means, 
                        yerr=[np.array(means) - np.array(cis_lower),
                              np.array(cis_upper) - np.array(means)],
                        fmt='o-', capsize=8, capthick=3, linewidth=3, 
                        markersize=10, color='blue', alpha=0.8)
            
            # 添加置信区间填充
            for i, (mean, lower, upper) in enumerate(zip(means, cis_lower, cis_upper)):
                ax4.fill_between([i-0.2, i+0.2], [lower, lower], [upper, upper], 
                               alpha=0.3, color='blue')
            
            ax4.set_xlabel('BMI组', fontweight='bold')
            ax4.set_ylabel('达标时间（周）', fontweight='bold')
            ax4.set_title('达标时间置信区间分析', fontweight='bold', fontsize=14)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(groups_ci)
            ax4.grid(True, alpha=0.3)
        
        # 5. 多维度性能对比 - 使用极坐标创建雷达图
        ax5 = fig.add_subplot(2, 3, 5, projection='polar')
        
        # 创建性能对比雷达图
        performance_metrics = ['达标率', '推荐时点', '风险因子（反向）']
        n_metrics = len(performance_metrics)
        
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        colors_perf = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for i, group_id in enumerate(groups):
            # 标准化性能指标到0-1
            reach_rate_norm = reach_rates[i]
            time_norm = 1 - (timepoints[i] - min(timepoints)) / (max(timepoints) - min(timepoints)) if max(timepoints) > min(timepoints) else 0.5
            risk_norm = 1 - (risk_factors[i] - min(risk_factors)) / (max(risk_factors) - min(risk_factors)) if max(risk_factors) > min(risk_factors) else 0.5
            
            values = [reach_rate_norm, time_norm, risk_norm, reach_rate_norm]  # 闭合
            
            ax5.plot(angles, values, 'o-', linewidth=2, 
                    label=f'组{group_id}', color=colors_perf[i], alpha=0.8)
            ax5.fill(angles, values, alpha=0.1, color=colors_perf[i])
        
        # 设置极坐标标签
        ax5.set_thetagrids(angles[:-1] * 180/np.pi, performance_metrics)
        ax5.set_ylim(0, 1)
        ax5.set_title('多维度性能对比雷达图', fontweight='bold', fontsize=14, pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax5.grid(True, alpha=0.3)
        
        # 6. 综合评分排序
        ax6 = fig.add_subplot(2, 3, 6)
        
        # 计算综合评分
        综合评分 = []
        for i in range(len(groups)):
            # 权重：达标率40%，时点合理性30%，风险控制30%
            reach_score = reach_rates[i] * 0.4
            time_score = (1 - abs(timepoints[i] - 15) / 10) * 0.3  # 15周为理想时点
            risk_score = (1 / risk_factors[i]) * 0.3 if risk_factors[i] > 0 else 0
            
            total_score = reach_score + time_score + risk_score
            综合评分.append(total_score)
        
        # 排序
        sorted_indices = np.argsort(综合评分)[::-1]  # 降序
        sorted_groups = [groups[i] for i in sorted_indices]
        sorted_scores = [综合评分[i] for i in sorted_indices]
        
        bars = ax6.barh(range(len(sorted_groups)), sorted_scores,
                       color=plt.cm.RdYlGn([score for score in sorted_scores]),
                       alpha=0.8, edgecolor='black', linewidth=1)
        
        ax6.set_yticks(range(len(sorted_groups)))
        ax6.set_yticklabels([f'BMI组{g}' for g in sorted_groups])
        ax6.set_xlabel('综合评分', fontweight='bold')
        ax6.set_title('BMI组综合性能排序', fontweight='bold', fontsize=14)
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 添加评分标签
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('picture/bmi_risk_assessment.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
    
    def create_factor_importance_visualization(self):
        """创建因素重要性可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 相关性因素重要性
        ax1 = axes[0, 0]
        if hasattr(self, 'correlation_matrix'):
            # 计算与目标变量的综合相关性
            time_corr = abs(self.correlation_matrix['事件时间'].drop(['事件时间', '事件观察']))
            event_corr = abs(self.correlation_matrix['事件观察'].drop(['事件时间', '事件观察']))
            combined_importance = (time_corr + event_corr) / 2
            combined_importance = combined_importance.sort_values(ascending=True).tail(10)
            
            bars = ax1.barh(range(len(combined_importance)), combined_importance.values, alpha=0.7)
            ax1.set_yticks(range(len(combined_importance)))
            ax1.set_yticklabels(combined_importance.index, fontsize=10)
            ax1.set_xlabel('平均相关性强度')
            ax1.set_title('因素重要性排序(相关性)', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, combined_importance.values)):
                ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
        
        # 2. 机器学习特征重要性
        ax2 = axes[0, 1]
        if hasattr(self, 'time_prediction_models') and '随机森林回归' in self.time_prediction_models:
            try:
                model = self.time_prediction_models['随机森林回归']
                feature_cols = self.significant_factors + ['多因素聚类组']
                importances = model.feature_importances_
                
                # 选择top10特征
                indices = np.argsort(importances)[-10:]
                selected_features = [feature_cols[i] for i in indices]
                selected_importances = importances[indices]
                
                bars = ax2.barh(range(len(selected_features)), selected_importances, alpha=0.7, color='green')
                ax2.set_yticks(range(len(selected_features)))
                ax2.set_yticklabels(selected_features, fontsize=10)
                ax2.set_xlabel('重要性得分')
                ax2.set_title('机器学习特征重要性', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                for i, (bar, val) in enumerate(zip(bars, selected_importances)):
                    ax2.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
            except:
                ax2.text(0.5, 0.5, '机器学习模型\n特征重要性\n(加载失败)', 
                        ha='center', va='center', transform=ax2.transAxes)
        
        # 3. 因素分布箱线图
        ax3 = axes[1, 0]
        if len(self.significant_factors) >= 3:
            factor_data = []
            factor_labels = []
            for factor in self.significant_factors[:6]:  # 选择前6个因素
                if factor in self.survival_df.columns:
                    data = self.survival_df[factor].dropna()
                    if len(data) > 0:
                        factor_data.append(data)
                        factor_labels.append(factor)
            
            if factor_data:
                bp = ax3.boxplot(factor_data, labels=factor_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                ax3.set_xlabel('因素')
                ax3.set_ylabel('标准化值')
                ax3.set_title('主要因素分布特征', fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
        
        # 4. 因素间交互作用热图
        ax4 = axes[1, 1]
        if hasattr(self, 'survival_df') and len(self.significant_factors) >= 4:
            # 计算主要因素间的交互相关性
            main_factors = self.significant_factors[:6]
            available_main_factors = [f for f in main_factors if f in self.survival_df.columns]
            
            if len(available_main_factors) >= 3:
                factor_subset = self.survival_df[available_main_factors].fillna(self.survival_df[available_main_factors].mean())
                interaction_matrix = factor_subset.corr()
                
                mask = np.triu(np.ones_like(interaction_matrix, dtype=bool))
                sns.heatmap(interaction_matrix, annot=True, cmap='coolwarm', center=0,
                           fmt='.2f', square=True, mask=mask, ax=ax4, cbar_kws={"shrink": .8})
                ax4.set_title('主要因素交互相关性', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, '因素数量不足\n无法生成交互图', 
                        ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig('picture/multifactor_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def create_survival_curves_visualization(self):
        """创建生存曲线可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 使用BMI分组而不是聚类组
        group_column = 'BMI分组' if 'BMI分组' in self.survival_df.columns else '多因素聚类组'
        
        # 1. 各组达标时间分布
        ax1 = axes[0, 0]
        observed_data = self.survival_df[self.survival_df['事件观察'] == 1]
        
        if len(observed_data) > 0:
            for group_id in sorted(self.survival_df[group_column].dropna().unique()):
                if group_id == -1:
                    continue
                group_observed = observed_data[observed_data[group_column] == group_id]
                if len(group_observed) > 5:  # 至少5个样本才画图
                    ax1.hist(group_observed['事件时间'], alpha=0.6, 
                            label=f'BMI组{group_id} (n={len(group_observed)})', 
                            bins=np.arange(10, 25, 1), density=True)
            
            ax1.set_xlabel('达标时间(周)')
            ax1.set_ylabel('密度')
            ax1.set_title('各BMI组达标时间分布', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 累积达标率曲线
        ax2 = axes[0, 1]
        for group_id in sorted(self.survival_df[group_column].dropna().unique()):
            if group_id == -1:
                continue
            group_data = self.survival_df[self.survival_df[group_column] == group_id]
            if len(group_data) > 5:
                times = np.arange(12, 22, 0.5)
                cum_rates = []
                for t in times:
                    reached = len(group_data[(group_data['事件观察'] == 1) & 
                                           (group_data['事件时间'] <= t)])
                    total = len(group_data)
                    cum_rates.append(reached / total)
                
                ax2.plot(times, cum_rates, marker='o', label=f'BMI组{group_id}', linewidth=2)
        
        ax2.set_xlabel('孕周')
        ax2.set_ylabel('累积达标率')
        ax2.set_title('累积达标率曲线', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. 检测次数与达标关系
        ax3 = axes[1, 0]
        detection_counts = self.survival_df['检测次数'].unique()
        reach_rates_by_detection = []
        detection_labels = []
        
        for count in sorted(detection_counts):
            if pd.notna(count):
                subset = self.survival_df[self.survival_df['检测次数'] == count]
                if len(subset) >= 3:  # 至少3个样本
                    reach_rate = subset['事件观察'].mean()
                    reach_rates_by_detection.append(reach_rate)
                    detection_labels.append(f'{int(count)}次')
        
        if reach_rates_by_detection:
            bars = ax3.bar(detection_labels, reach_rates_by_detection, alpha=0.7, color='skyblue')
            ax3.set_xlabel('检测次数')
            ax3.set_ylabel('达标率')
            ax3.set_title('检测次数与达标率关系', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            for bar, rate in zip(bars, reach_rates_by_detection):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
        
        # 4. BMI与年龄联合效应
        ax4 = axes[1, 1]
        if 'BMI' in self.survival_df.columns and '年龄' in self.survival_df.columns:
            # 创建BMI-年龄网格
            bmi_bins = pd.cut(self.survival_df['BMI'], bins=5, labels=False)
            age_bins = pd.cut(self.survival_df['年龄'], bins=4, labels=False)
            
            pivot_data = []
            for i in range(5):
                row_data = []
                for j in range(4):
                    mask = (bmi_bins == i) & (age_bins == j)
                    if mask.sum() > 0:
                        reach_rate = self.survival_df.loc[mask, '事件观察'].mean()
                        row_data.append(reach_rate)
                    else:
                        row_data.append(np.nan)
                pivot_data.append(row_data)
            
            pivot_df = pd.DataFrame(pivot_data)
            
            im = ax4.imshow(pivot_df, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax4.set_xlabel('年龄分组')
            ax4.set_ylabel('BMI分组')
            ax4.set_title('BMI-年龄联合效应热图', fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('达标率')
        
        plt.tight_layout()
        plt.savefig('picture/bmi_survival_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n" + "="*80)
        print("问题3: 多因素综合风险评估与最优检测时点 - 分析报告")
        print("="*80)
        
        print("\n🔬 分析方法学创新:")
        print("1. 多维聚类分析：K-means + 层次聚类 + DBSCAN密度聚类")
        print("2. 机器学习预测：随机森林 + 梯度提升 + 神经网络")
        print("3. 蒙特卡洛仿真：检测误差影响量化评估")
        print("4. 综合风险函数：早期/晚期/未达标/误差多重风险平衡")
        
        print(f"\n📊 多因素分析结果:")
        if hasattr(self, 'significant_factors'):
            print(f"显著影响因素: {', '.join(self.significant_factors)}")
        
        if hasattr(self, 'survival_df'):
            total_samples = len(self.survival_df)
            total_reached = self.survival_df['事件观察'].sum()
            overall_reach_rate = total_reached / total_samples
            print(f"总样本数: {total_samples}")
            print(f"总达标数: {total_reached}")
            print(f"总体达标率: {overall_reach_rate:.1%}")
        
        print(f"\n🎯 基于BMI的多因素优化分组与推荐时点:")
        print("-" * 80)
        
        if hasattr(self, 'multifactor_recommendations'):
            for _, rec in self.multifactor_recommendations.iterrows():
                print(f"BMI组{rec['BMI组ID']}: {rec['组描述']}")
                print(f"  样本特征: {rec['样本数量']}个样本, 达标率{rec['达标率']}")
                print(f"  平均达标时间: {rec['平均达标时间']}")
                print(f"  综合风险因子: {rec['综合风险因子']}")
                print(f"  检测误差影响: {rec['检测误差影响']}")
                print(f"  🎯 推荐检测时点: {rec['推荐检测时点']}")
                print()
        
        print(f"\n🛡️ 检测误差影响评估:")
        if hasattr(self, 'error_simulation_results'):
            for group_id, error_stats in self.error_simulation_results.items():
                print(f"BMI组{group_id}:")
                print(f"  假阳性率: {error_stats['false_positive_rate']:.3f}")
                print(f"  假阴性率: {error_stats['false_negative_rate']:.3f}")
                print(f"  达标率置信区间: [{error_stats['reach_rate_ci_lower']:.3f}, {error_stats['reach_rate_ci_upper']:.3f}]")
                print(f"  时间置信区间: [{error_stats['time_ci_lower']:.1f}, {error_stats['time_ci_upper']:.1f}]周")
        
        print(f"\n🧠 机器学习模型表现:")
        if hasattr(self, 'time_prediction_models'):
            print("达标时间预测模型:")
            for model_name in self.time_prediction_models.keys():
                print(f"  ✅ {model_name}: 已训练完成")
        
        if hasattr(self, 'reach_prediction_models'):
            print("达标概率预测模型:")
            for model_name in self.reach_prediction_models.keys():
                print(f"  ✅ {model_name}: 已训练完成")
        
        print(f"\n💡 核心创新点:")
        print("1. 明确的BMI区间分组：解决了重叠区间问题，便于临床应用")
        print("2. 多因素综合优化：在BMI分组基础上考虑年龄、检测复杂性等")
        print("3. 机器学习预测：构建多种模型预测达标时间和概率")
        print("4. 误差量化评估：蒙特卡洛仿真量化检测误差影响")
        print("5. 综合风险优化：构建多维风险函数寻找最优时点")
        
        print(f"\n🔬 科学性与可靠性:")
        print("✅ 基于BMI的明确分组（无重叠区间）")
        print("✅ 多因素风险评分系统")
        print("✅ 机器学习模型交叉验证")
        print("✅ 蒙特卡洛仿真误差评估")
        print("✅ 综合风险函数多目标优化")
        print("✅ 置信区间和不确定性量化")
        
        # 保存结果
        if hasattr(self, 'multifactor_recommendations'):
            self.multifactor_recommendations.to_excel('bmi_multifactor_nipt_recommendations.xlsx', index=False)
            print(f"\n✅ BMI分组多因素优化推荐方案已保存至: bmi_multifactor_nipt_recommendations.xlsx")
        
        return True
    
    def run_complete_analysis(self):
        """运行完整的多因素分析"""
        print("NIPT问题3: 多因素综合风险评估与最优检测时点")
        print("="*80)
        
        # 1. 数据加载与预处理
        if not self.load_and_process_data():
            return False
        
        # 2. 多因素相关性分析
        if not self.multifactor_correlation_analysis():
            print("⚠️ 多因素相关性分析失败")
            return False
        
        # 3. 基于BMI的分组分析
        if not self.bmi_based_grouping_with_multifactor_optimization():
            print("⚠️ BMI分组分析失败")
            return False
        
        # 4. 机器学习风险预测
        if not self.machine_learning_risk_prediction():
            print("⚠️ 机器学习模型训练失败，继续其他分析")
        
        # 5. 蒙特卡洛误差仿真
        if not self.monte_carlo_error_simulation():
            print("⚠️ 误差仿真失败，继续其他分析")
        
        # 6. 综合风险优化
        self.comprehensive_risk_optimization()
        
        # 7. 创建分离的可视化图表
        print("\n生成可视化图表...")
        self.create_clustering_visualization()
        self.create_risk_assessment_visualization()
        self.create_factor_importance_visualization()
        self.create_survival_curves_visualization()
        
        # 8. 生成报告
        self.generate_comprehensive_report()
        
        return True

# 主程序执行
if __name__ == "__main__":
    analyzer = MultifactorNIPTOptimizer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n" + "="*80)
        print("🎉 问题3：多因素综合风险评估与最优检测时点分析完成！")
        print("="*80)
        print("核心创新:")
        print("✅ 基于BMI的明确分组（解决区间重叠问题）")
        print("✅ 多因素综合风险评估（年龄+检测复杂性+浓度风险）") 
        print("✅ 蒙特卡洛误差仿真（量化检测误差影响）")
        print("✅ 综合风险函数优化（多目标平衡最优解）")
        print("✅ 不确定性量化评估（置信区间分析）")
        print("="*80)
    else:
        print("❌ 分析失败，请检查数据文件和依赖环境")
