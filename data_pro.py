import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import re  # 用于孕周转换的正则验证
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataPreprocessorCorrect:
    def __init__(self, file_path, sheet_name=None):
        self.file_path = file_path
        self.sheet_name = sheet_name  # 新增：支持指定工作表
        self.data = None
        self.original_data = None
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=5)  # 新增：KNN 插值器，用于缺失值
        self.label_encoders = {}
        
    def load_data(self):
        """加载数据，支持指定工作表，并添加预览和类型检查"""
        try:
            # 先检查Excel文件的工作表
            excel_file = pd.ExcelFile(self.file_path)
            print(f"Excel文件中的工作表: {excel_file.sheet_names}")
            
            # 如果没有指定工作表，使用第一个工作表
            if self.sheet_name is None:
                self.sheet_name = excel_file.sheet_names[0]
                print(f"未指定工作表，使用第一个工作表: {self.sheet_name}")
            
            # 加载指定的工作表
            self.data = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            self.original_data = self.data.copy()
            print(f"数据加载成功，形状: {self.data.shape}")
            print(f"列名前10个: {list(self.data.columns[:10])}")
            print("\n数据预览（前5行）：")
            print(self.data.head())
            print("\n数据类型：")
            print(self.data.dtypes)
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def convert_gestational_week(self, week_str):
        """
        将孕周转换为十进制格式，使用正则验证，并添加范围检查
        '13w+6' -> 13.857 (13 + 6/7)
        '12w' -> 12.0
        范围: 9-40 周，超出返回 np.nan
        """
        if pd.isna(week_str):
            return np.nan
        week_str = str(week_str).strip().lower()
        pattern = r'^(\d+)(w\+(\d+))?$|^(\d+)(w)?$'
        match = re.match(pattern, week_str)
        if not match:
            return np.nan
        try:
            if match.group(2):  # 格式如 "13w+6"
                weeks = float(match.group(1))
                days = float(match.group(3))
                result = round(weeks + days / 7, 3)
            else:  # 格式如 "13w" 或 "13"
                result = float(match.group(4) or match.group(1))
            # 新增：范围检查（NIPT 典型范围 9-40 周）
            if result < 9 or result > 40:
                return np.nan
            return result
        except:
            return np.nan
    
    def clean_and_prepare_data(self):
        """数据清洗和准备，添加缺失比例检查和异常值截断"""
        print("\n" + "="*60)
        print("数据清洗和准备")
        print("="*60)
        
        # 1. 孕周转换
        if '检测孕周' in self.data.columns:
            print("转换孕周格式...")
            self.data['孕周_数值'] = self.data['检测孕周'].apply(self.convert_gestational_week)
            
            # 显示转换示例
            sample_data = self.data[['检测孕周', '孕周_数值']].dropna().head(10)
            print("孕周转换示例:")
            for _, row in sample_data.iterrows():
                print(f"{row['检测孕周']} -> {row['孕周_数值']}")
            
            print(f"孕周转换完成，缺失值: {self.data['孕周_数值'].isnull().sum()}")
        
        # 2. 处理胎儿健康状态，新增默认值处理
        if '胎儿是否健康' in self.data.columns:
            health_map = {'是': 1, '否': 0, '未知': np.nan}  # 新增未知处理
            self.data['胎儿健康_数值'] = self.data['胎儿是否健康'].map(health_map).fillna(np.nan)
            print("胎儿健康状态转换完成")
        
        # 3. 处理分类变量编码
        print("\n处理分类变量...")
        
        # IVF妊娠，新增缺失比例检查
        if 'IVF妊娠' in self.data.columns:
            missing_ratio = self.data['IVF妊娠'].isnull().mean()
            print(f"IVF妊娠缺失比例: {missing_ratio:.2%}")
            if missing_ratio > 0.1:  # 如果缺失>10%，填充为'未知'
                self.data['IVF妊娠'] = self.data['IVF妊娠'].fillna('未知')
            else:
                self.data['IVF妊娠'] = self.data['IVF妊娠'].fillna('自然受孕')
            
            ivf_map = {'自然受孕': 0, 'IVF（试管婴儿）': 1, 'IUI（人工授精）': 2, '未知': -1}
            unique_vals = self.data['IVF妊娠'].unique()
            for val in unique_vals:
                if val not in ivf_map:
                    if '试管' in str(val) or 'IVF' in str(val):
                        ivf_map[val] = 1
                    elif '人工' in str(val) or 'IUI' in str(val):
                        ivf_map[val] = 2
                    else:
                        ivf_map[val] = 0
            
            self.data['IVF妊娠_编码'] = self.data['IVF妊娠'].map(ivf_map)
            print(f"IVF妊娠编码: {ivf_map}")
        
        # 4. 创建分组变量并编码，调整 BMI 分箱为中国人群标准
        print("\n创建分组变量...")
        
        # BMI分组（针对孕妇群体的合理分组）
        if '孕妇BMI' in self.data.columns:
            # 先检查BMI分布，确定合理的分组点
            bmi_data = self.data['孕妇BMI'].dropna()
            bmi_quartiles = bmi_data.quantile([0.25, 0.5, 0.75])
            print(f"BMI分布: Q1={bmi_quartiles[0.25]:.1f}, 中位数={bmi_quartiles[0.5]:.1f}, Q3={bmi_quartiles[0.75]:.1f}")
            
            def bmi_group(bmi):
                if pd.isna(bmi): return 0
                elif bmi < 25.0: return 1   # 正常体重（合并偏瘦和正常，适合孕妇）
                elif bmi < 28.0: return 2   # 超重（孕妇超重标准适当放宽）
                elif bmi < 32.0: return 3   # 轻度肥胖
                elif bmi < 36.0: return 4   # 中度肥胖
                else: return 5              # 重度肥胖
            
            self.data['BMI分组_编码'] = self.data['孕妇BMI'].apply(bmi_group)
            print("BMI分组: 0=未知, 1=正常(<25), 2=超重(25-28), 3=轻度肥胖(28-32), 4=中度肥胖(32-36), 5=重度肥胖(≥36)")
            
            # 显示各组样本量
            group_counts = self.data['BMI分组_编码'].value_counts().sort_index()
            print("各BMI组样本量:", dict(group_counts))
        
        # 年龄分组
        if '年龄' in self.data.columns:
            def age_group(age):
                if pd.isna(age): return 0
                elif age < 25: return 1  # 20-25岁
                elif age < 30: return 2  # 26-30岁
                elif age < 35: return 3  # 31-35岁
                else: return 4  # 35岁以上
            
            self.data['年龄分组_编码'] = self.data['年龄'].apply(age_group)
            print("年龄分组: 0=未知, 1=20-25岁, 2=26-30岁, 3=31-35岁, 4=35岁以上")
        
        # 孕期阶段分组
        if '孕周_数值' in self.data.columns:
            def pregnancy_stage(week):
                if pd.isna(week): return 0
                elif week < 14: return 1  # 孕早期
                elif week < 21: return 2  # 孕中期
                else: return 3  # 孕晚期
            
            self.data['孕期阶段_编码'] = self.data['孕周_数值'].apply(pregnancy_stage)
            print("孕期阶段: 0=未知, 1=孕早期, 2=孕中期, 3=孕晚期")
        
        # 5. 处理身高体重数据清洗和标准化
        print("\n处理身高体重数据...")
        
        # 处理身高数据
        if '身高' in self.data.columns:
            print("清洗身高数据...")
            # 去除异常值和标准化身高（正常范围：140-200cm）
            height_before = self.data['身高'].notna().sum()
            self.data['身高'] = pd.to_numeric(self.data['身高'], errors='coerce')
            self.data['身高'] = self.data['身高'].where(
                (self.data['身高'] >= 140) & (self.data['身高'] <= 200), np.nan
            )
            height_after = self.data['身高'].notna().sum()
            print(f"身高数据: {height_before} -> {height_after} (去除异常值)")
            
            # 身高分组
            def height_group(height):
                if pd.isna(height): return 0
                elif height < 155: return 1  # 偏矮
                elif height < 165: return 2  # 中等
                else: return 3               # 偏高
            
            self.data['身高分组_编码'] = self.data['身高'].apply(height_group)
            print("身高分组: 0=未知, 1=偏矮(<155cm), 2=中等(155-165cm), 3=偏高(≥165cm)")
        
        # 处理体重数据
        if '体重' in self.data.columns:
            print("清洗体重数据...")
            # 去除异常值和标准化体重（正常范围：40-120kg，考虑孕妇）
            weight_before = self.data['体重'].notna().sum()
            self.data['体重'] = pd.to_numeric(self.data['体重'], errors='coerce')
            self.data['体重'] = self.data['体重'].where(
                (self.data['体重'] >= 40) & (self.data['体重'] <= 120), np.nan
            )
            weight_after = self.data['体重'].notna().sum()
            print(f"体重数据: {weight_before} -> {weight_after} (去除异常值)")
            
            # 体重分组（针对孕妇的合理分组）
            def weight_group(weight):
                if pd.isna(weight): return 0
                elif weight < 55: return 1   # 偏轻
                elif weight < 70: return 2   # 正常
                elif weight < 85: return 3   # 偏重
                else: return 4               # 超重
            
            self.data['体重分组_编码'] = self.data['体重'].apply(weight_group)
            print("体重分组: 0=未知, 1=偏轻(<55kg), 2=正常(55-70kg), 3=偏重(70-85kg), 4=超重(≥85kg)")
        
        # 如果有身高体重数据但没有BMI，计算BMI
        if '身高' in self.data.columns and '体重' in self.data.columns and '孕妇BMI' not in self.data.columns:
            print("根据身高体重计算BMI...")
            # BMI = 体重(kg) / (身高(m))^2
            self.data['孕妇BMI'] = self.data['体重'] / ((self.data['身高'] / 100) ** 2)
            bmi_count = self.data['孕妇BMI'].notna().sum()
            print(f"成功计算BMI: {bmi_count}个样本")
        
        # 6. 处理染色体非整倍体，新增缺失比例检查
        if '染色体的非整倍体' in self.data.columns:
            missing_ratio = self.data['染色体的非整倍体'].isnull().mean()
            print(f"染色体非整倍体缺失比例: {missing_ratio:.2%}")
            if missing_ratio > 0.1:
                self.data['染色体的非整倍体'] = self.data['染色体的非整倍体'].fillna('未知')
            else:
                self.data['染色体的非整倍体'] = self.data['染色体的非整倍体'].fillna('正常')
            
            normal_vals = ['正常', 'Normal', '无', 'None']
            self.data['染色体异常_编码'] = self.data['染色体的非整倍体'].apply(
                lambda x: 0 if str(x) in normal_vals else -1 if str(x) == '未知' else 1
            )
            print("染色体异常编码: 0=正常, 1=异常, -1=未知")
        
        # 6. 处理关键字段缺失，使用 KNN 插值代替直接删除（如果缺失比例不高）
        key_columns = ['年龄', '孕妇BMI', '孕周_数值', 'Y染色体浓度', '胎儿健康_数值', '身高', '体重']
        available_key_cols = [col for col in key_columns if col in self.data.columns]
        
        if available_key_cols:
            print("\n关键字段缺失比例检查:")
            for col in available_key_cols:
                missing_ratio = self.data[col].isnull().mean()
                print(f"{col}: {missing_ratio:.2%}")
            
            before_impute = self.data[available_key_cols].isnull().sum().sum()
            if before_impute > 0:
                print(f"使用 KNN 插值填充缺失值...")
                self.data[available_key_cols] = pd.DataFrame(
                    self.imputer.fit_transform(self.data[available_key_cols]),
                    columns=available_key_cols,
                    index=self.data.index
                )
            after_impute = self.data[available_key_cols].isnull().sum().sum()
            print(f"插值后剩余缺失: {after_impute}")
        
        # 7. 新增：异常值截断
        print("\n异常值截断...")
        if '孕妇BMI' in self.data.columns:
            self.data['孕妇BMI'] = np.clip(self.data['孕妇BMI'], 15, 50)
        if '孕周_数值' in self.data.columns:
            self.data['孕周_数值'] = np.clip(self.data['孕周_数值'], 9, 40)
        if '年龄' in self.data.columns:
            self.data['年龄'] = np.clip(self.data['年龄'], 18, 50)  # 假设孕妇年龄范围
        if '身高' in self.data.columns:
            self.data['身高'] = np.clip(self.data['身高'], 140, 200)  # 身高合理范围
        if '体重' in self.data.columns:
            self.data['体重'] = np.clip(self.data['体重'], 40, 120)   # 孕妇体重合理范围
        print("异常值截断完成")
    
    def create_derived_features(self):
        """创建衍生特征，新增交互项和异常值处理"""
        print("\n" + "="*60)
        print("创建衍生特征")
        print("="*60)
        
        # 1. 染色体Z值综合指标
        z_columns = [col for col in self.data.columns if 'Z值' in col and '染色体' in col]
        if z_columns:
            self.data['Z值异常度'] = self.data[z_columns].abs().sum(axis=1)
            # 新增：截断异常值（99% 分位数）
            self.data['Z值异常度'] = np.clip(self.data['Z值异常度'], 0, self.data['Z值异常度'].quantile(0.99))
            print(f"创建Z值异常度特征，使用列: {z_columns}")
        
        # 2. GC含量综合指标
        gc_columns = [col for col in self.data.columns if 'GC含量' in col and '染色体' in col]
        if gc_columns:
            self.data['GC含量均值'] = self.data[gc_columns].mean(axis=1)
            self.data['GC含量标准差'] = self.data[gc_columns].std(axis=1)
            print(f"创建GC含量综合特征，使用列: {gc_columns}")
        
        # 3. BMI * 年龄交互项
        if '孕妇BMI' in self.data.columns and '年龄' in self.data.columns:
            self.data['BMI年龄交互'] = self.data['孕妇BMI'] * self.data['年龄']
            print("创建BMI-年龄交互特征")
        
        # 4. 孕周平方项
        if '孕周_数值' in self.data.columns:
            self.data['孕周平方'] = self.data['孕周_数值'] ** 2
            print("创建孕周平方特征")
        
        # 4. 新增：身高 * 体重交互项
        if '身高' in self.data.columns and '体重' in self.data.columns:
            self.data['身高体重交互'] = self.data['身高'] * self.data['体重']
            print("创建身高-体重交互特征")
        
        # 5. 新增：孕周 * BMI 交互项（探索更多关系）
        if '孕周_数值' in self.data.columns and '孕妇BMI' in self.data.columns:
            self.data['孕周BMI交互'] = self.data['孕周_数值'] * self.data['孕妇BMI']
            print("创建孕周-BMI交互特征")
    
    def prepare_final_dataset(self):
        """准备最终的建模数据集，新增偏态检查和变换，确保目标不进入X"""
        print("\n" + "="*60)
        print("准备最终建模数据集")
        print("="*60)
        
        # 选择数值型特征进行标准化（排除目标变量）
        numeric_features = [
            '年龄', '孕妇BMI', '孕周_数值', '身高', '体重'  # 排除 'Y染色体浓度'，添加身高体重
        ]
        
        # 添加可用的衍生数值特征
        derived_numeric = ['Z值异常度', 'GC含量均值', 'GC含量标准差', 'BMI年龄交互', '孕周平方', '孕周BMI交互', '身高体重交互']
        for col in derived_numeric:
            if col in self.data.columns:
                numeric_features.append(col)
        
        # 添加可用的染色体指标
        z_columns = [col for col in self.data.columns if 'Z值' in col and '染色体' in col]
        numeric_features.extend(z_columns)
        
        # GC含量相关
        if 'GC含量' in self.data.columns:
            numeric_features.append('GC含量')
        
        # 过滤实际存在的列
        numeric_features = [col for col in numeric_features if col in self.data.columns]
        print(f"数值型特征 ({len(numeric_features)}个): {numeric_features}")
        
        # 新增：检查偏态并应用对数变换
        print("\n检查数值特征偏态...")
        for col in numeric_features:
            skewness = self.data[col].skew()
            if abs(skewness) > 1:
                print(f"{col} 偏态: {skewness:.2f}，应用对数变换")
                self.data[col] = np.log1p(self.data[col].clip(lower=0))  # 避免负值
        
        # 分类特征（已编码）
        categorical_features = [
            'IVF妊娠_编码', 'BMI分组_编码', '年龄分组_编码', 
            '孕期阶段_编码', '染色体异常_编码', '身高分组_编码', '体重分组_编码'
        ]
        categorical_features = [col for col in categorical_features if col in self.data.columns]
        print(f"分类特征 ({len(categorical_features)}个): {categorical_features}")
        
        # 创建建模数据集
        modeling_features = numeric_features + categorical_features
        
        # 检查数据
        print(f"\n数据检查:")
        for feature in modeling_features:
            non_null_count = self.data[feature].notna().sum()
            print(f"{feature}: {non_null_count}/{len(self.data)} 非空值")
        
        # 创建特征矩阵X
        X = self.data[modeling_features].copy()
        
        # 标准化数值特征
        if numeric_features:
            print(f"\n标准化数值特征...")
            X_numeric = X[numeric_features].copy()
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            X_numeric_scaled = pd.DataFrame(X_numeric_scaled, 
                                          columns=numeric_features, 
                                          index=X.index)
            
            # 替换原始数值特征
            for col in numeric_features:
                X[col] = X_numeric_scaled[col]
            
            print("数值特征标准化完成")
        
        # 目标变量
        y_concentration = None
        y_health = None
        
        if 'Y染色体浓度' in self.data.columns:
            y_concentration = self.data['Y染色体浓度'].copy()
            print(f"Y染色体浓度目标变量: {len(y_concentration)} 个样本")
        
        if '胎儿健康_数值' in self.data.columns:
            y_health = self.data['胎儿健康_数值'].copy()
            healthy_count = y_health.sum() if y_health is not None else 0
            unhealthy_count = len(y_health) - healthy_count if y_health is not None else 0
            print(f"胎儿健康目标变量: 健康={healthy_count}, 不健康={unhealthy_count}")
        
        return {
            'X': X,
            'y_concentration': y_concentration,
            'y_health': y_health,
            'feature_names': modeling_features,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
    
    def save_clean_data(self, modeling_data, output_path='cleaned_data.xlsx'):
        """保存清洁的建模数据，新增保存前完整性检查"""
        print(f"\n保存清洁数据到: {output_path}")
        
        # 创建最终数据集
        final_data = modeling_data['X'].copy()
        
        if modeling_data['y_concentration'] is not None:
            final_data['Y染色体浓度_目标'] = modeling_data['y_concentration']
        
        if modeling_data['y_health'] is not None:
            final_data['胎儿健康_目标'] = modeling_data['y_health']
        
        # 新增：保存前检查缺失值
        if final_data.isnull().any().any():
            print("警告：保存的数据中存在缺失值！请检查插值步骤")
        else:
            print("数据完整，无缺失值")
        
        # 保存，处理权限错误
        try:
            final_data.to_excel(output_path, index=False)
            print(f"数据成功保存到: {output_path}")
        except PermissionError:
            # 如果文件被占用，尝试使用不同的文件名
            import time
            timestamp = int(time.time())
            backup_path = f'cleaned_data_backup_{timestamp}.xlsx'
            print(f"原文件被占用，保存到备份文件: {backup_path}")
            final_data.to_excel(backup_path, index=False)
            print(f"数据成功保存到: {backup_path}")
        except Exception as e:
            print(f"保存Excel失败: {e}")
            # 尝试保存为CSV格式
            csv_path = output_path.replace('.xlsx', '.csv')
            print(f"尝试保存为CSV格式: {csv_path}")
            final_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"数据成功保存到: {csv_path}")
        
        # 显示最终数据信息
        print(f"最终数据形状: {final_data.shape}")
        print(f"特征列: {len(modeling_data['feature_names'])}")
        print(f"数值特征: {len(modeling_data['numeric_features'])}")
        print(f"分类特征: {len(modeling_data['categorical_features'])}")
        
        # 显示前几行
        print(f"\n最终数据预览:")
        print(final_data.head())
        
        # 检查是否还有文字数据
        print(f"\n数据类型检查:")
        for col in final_data.columns:
            dtype = final_data[col].dtype
            if dtype == 'object':
                print(f"警告: {col} 仍为object类型")
            else:
                print(f"{col}: {dtype}")
    
    def run_complete_preprocessing(self):
        """运行完整预处理流程"""
        print("="*80)
        print("开始完整数据预处理")
        print("="*80)
        
        # 1. 加载数据
        if not self.load_data():
            return None
        
        # 2. 数据清洗和准备
        self.clean_and_prepare_data()
        
        # 3. 创建衍生特征
        self.create_derived_features()
        
        # 4. 准备最终数据集
        modeling_data = self.prepare_final_dataset()
        
        # 5. 保存清洁数据
        self.save_clean_data(modeling_data)
        
        print("\n" + "="*80)
        print("数据预处理完成！")
        print("="*80)
        
        return {
            'modeling_data': modeling_data,
            'original_data': self.original_data,
            'processed_data': self.data,
            'scaler': self.scaler
        }

# 使用示例
if __name__ == "__main__":
    # 创建预处理器，支持指定工作表
    preprocessor = DataPreprocessorCorrect('附件.xlsx', sheet_name=None)  # 如果有多张表，指定 sheet_name='Sheet1'
    
    # 运行完整预处理
    results = preprocessor.run_complete_preprocessing()
    
    if results:
        modeling_data = results['modeling_data']
        X = modeling_data['X']
        y_concentration = modeling_data['y_concentration']
        y_health = modeling_data['y_health']
        
        print(f"\n" + "="*50)
        print("最终建模数据总结")
        print("="*50)
        print(f"特征矩阵X形状: {X.shape}")
        if y_concentration is not None:
            print(f"Y染色体浓度范围: [{y_concentration.min():.4f}, {y_concentration.max():.4f}]")
            print(f"Y染色体浓度均值: {y_concentration.mean():.4f}")
        if y_health is not None:
            print(f"胎儿健康分布: {y_health.value_counts().to_dict()}")
        
        print(f"\n特征列表:")
        for i, feature in enumerate(modeling_data['feature_names'], 1):
            print(f"{i:2d}. {feature}")
