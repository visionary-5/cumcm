import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataPreprocessorCorrect:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.original_data = None
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_excel(self.file_path)
            self.original_data = self.data.copy()
            print(f"数据加载成功，形状: {self.data.shape}")
            print(f"列名前10个: {list(self.data.columns[:10])}")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def convert_gestational_week(self, week_str):
        """
        将孕周转换为十进制格式
        '13w+6' -> 13.857 (13 + 6/7)
        '12w' -> 12.0
        """
        if pd.isna(week_str):
            return np.nan
        
        week_str = str(week_str).strip()
        try:
            if 'w' in week_str.lower():
                if '+' in week_str:
                    # 处理 '13w+6' 格式
                    parts = week_str.lower().split('w+')
                    weeks = float(parts[0])
                    days = float(parts[1])
                    return round(weeks + days / 7, 3)  # 保留3位小数
                else:
                    # 处理 '13w' 格式
                    week_part = week_str.lower().replace('w', '')
                    return float(week_part)
            else:
                return float(week_str)
        except:
            return np.nan
    
    def clean_and_prepare_data(self):
        """数据清洗和准备"""
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
        
        # 2. 处理胎儿健康状态
        if '胎儿是否健康' in self.data.columns:
            health_map = {'是': 1, '否': 0}
            self.data['胎儿健康_数值'] = self.data['胎儿是否健康'].map(health_map)
            print("胎儿健康状态转换完成")
        
        # 3. 处理分类变量编码
        print("\n处理分类变量...")
        
        # IVF妊娠
        if 'IVF妊娠' in self.data.columns:
            self.data['IVF妊娠'] = self.data['IVF妊娠'].fillna('自然受孕')
            ivf_map = {'自然受孕': 0, 'IVF（试管婴儿）': 1, 'IUI（人工授精）': 2}
            # 处理可能的其他值
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
        
        # 4. 创建分组变量并编码
        print("\n创建分组变量...")
        
        # BMI分组
        if '孕妇BMI' in self.data.columns:
            def bmi_group(bmi):
                if pd.isna(bmi): return 0
                elif bmi < 18.5: return 1  # 偏瘦
                elif bmi < 24.0: return 2  # 正常
                elif bmi < 28.0: return 3  # 超重
                else: return 4  # 肥胖
            
            self.data['BMI分组_编码'] = self.data['孕妇BMI'].apply(bmi_group)
            print("BMI分组: 0=未知, 1=偏瘦, 2=正常, 3=超重, 4=肥胖")
        
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
        
        # 5. 处理染色体非整倍体
        if '染色体的非整倍体' in self.data.columns:
            self.data['染色体的非整倍体'] = self.data['染色体的非整倍体'].fillna('正常')
            # 简单编码：正常=0，异常=1
            normal_vals = ['正常', 'Normal', '无', 'None']
            self.data['染色体异常_编码'] = self.data['染色体的非整倍体'].apply(
                lambda x: 0 if str(x) in normal_vals else 1
            )
            print("染色体异常编码: 0=正常, 1=异常")
        
        # 6. 删除关键字段缺失的样本
        key_columns = ['年龄', '孕妇BMI', '孕周_数值', 'Y染色体浓度', '胎儿健康_数值']
        available_key_cols = [col for col in key_columns if col in self.data.columns]
        
        if available_key_cols:
            before_drop = len(self.data)
            self.data = self.data.dropna(subset=available_key_cols)
            after_drop = len(self.data)
            print(f"\n删除关键字段缺失样本: {before_drop - after_drop} 条")
            print(f"剩余样本数: {after_drop}")
    
    def create_derived_features(self):
        """创建衍生特征"""
        print("\n" + "="*60)
        print("创建衍生特征")
        print("="*60)
        
        # 1. 染色体Z值综合指标
        z_columns = [col for col in self.data.columns if 'Z值' in col and '染色体' in col]
        if z_columns:
            # Z值异常度：所有Z值绝对值之和
            self.data['Z值异常度'] = self.data[z_columns].abs().sum(axis=1)
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
        
        # 4. 孕周平方项（可能存在非线性关系）
        if '孕周_数值' in self.data.columns:
            self.data['孕周平方'] = self.data['孕周_数值'] ** 2
            print("创建孕周平方特征")
    
    def prepare_final_dataset(self):
        """准备最终的建模数据集"""
        print("\n" + "="*60)
        print("准备最终建模数据集")
        print("="*60)
        
        # 选择数值型特征进行标准化
        numeric_features = [
            '年龄', '孕妇BMI', '孕周_数值', 'Y染色体浓度'
        ]
        
        # 添加可用的衍生数值特征
        derived_numeric = ['Z值异常度', 'GC含量均值', 'GC含量标准差', 'BMI年龄交互', '孕周平方']
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
        
        # 分类特征（已编码）
        categorical_features = [
            'IVF妊娠_编码', 'BMI分组_编码', '年龄分组_编码', 
            '孕期阶段_编码', '染色体异常_编码'
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
        """保存清洁的建模数据"""
        print(f"\n保存清洁数据到: {output_path}")
        
        # 创建最终数据集
        final_data = modeling_data['X'].copy()
        
        if modeling_data['y_concentration'] is not None:
            final_data['Y染色体浓度_目标'] = modeling_data['y_concentration']
        
        if modeling_data['y_health'] is not None:
            final_data['胎儿健康_目标'] = modeling_data['y_health']
        
        # 保存
        final_data.to_excel(output_path, index=False)
        
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
    # 创建预处理器
    preprocessor = DataPreprocessorCorrect('附件.xlsx')
    
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