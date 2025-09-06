# NIPT问题1：胎儿Y染色体浓度与孕妇孕周数和BMI等指标的相关特性分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency, jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-v0_8')

# 导入数据预处理类
from data_pro import DataPreprocessorCorrect

class NIPTAnalyzer:
    """NIPT数据分析类 - 专注于问题1的分析"""
    
    def __init__(self, data_file='附件.xlsx'):
        self.data_file = data_file
        self.cleaned_data = None
        self.male_data = None  # 男胎数据
        self.correlation_results = {}
        self.model_results = {}
        
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print("="*80)
        print("开始数据预处理")
        print("="*80)
        
        # 运行数据预处理
        preprocessor = DataPreprocessorCorrect(self.data_file, sheet_name='男胎检测数据')
        results = preprocessor.run_complete_preprocessing()
        
        if results:
            # 获取预处理后的数据
            processed_data = results['processed_data']
            
            # 创建分析数据集，包含原始值和编码值
            self.cleaned_data = pd.DataFrame({
                # 基础变量
                '年龄': processed_data['年龄'],
                '孕妇BMI': processed_data['孕妇BMI'],
                '孕周_数值': processed_data['孕周_数值'],
                'Y染色体浓度': processed_data['Y染色体浓度'],
                '胎儿健康': processed_data['胎儿健康_数值'],
                
                # 分组编码
                'BMI分组': processed_data['BMI分组_编码'],
                '年龄分组': processed_data['年龄分组_编码'], 
                '孕期阶段': processed_data['孕期阶段_编码'],
                'IVF妊娠': processed_data['IVF妊娠_编码'],
                
                # 衍生特征
                'BMI年龄交互': processed_data['BMI年龄交互'],
                '孕周平方': processed_data['孕周平方'],
                '孕周BMI交互': processed_data['孕周BMI交互'],
                
                # 染色体指标
                '13号染色体Z值': processed_data['13号染色体的Z值'],
                '18号染色体Z值': processed_data['18号染色体的Z值'],
                '21号染色体Z值': processed_data['21号染色体的Z值'],
                'X染色体Z值': processed_data['X染色体的Z值'],
                'Y染色体Z值': processed_data['Y染色体的Z值'],
            })
            
            # 过滤男胎数据（Y染色体浓度>0的记录）
            self.male_data = self.cleaned_data[self.cleaned_data['Y染色体浓度'] > 0].copy()
            
            print(f"\n数据加载完成:")
            print(f"总样本数: {len(self.cleaned_data)}")
            print(f"男胎样本数: {len(self.male_data)}")
            print(f"Y染色体浓度范围: [{self.male_data['Y染色体浓度'].min():.4f}, {self.male_data['Y染色体浓度'].max():.4f}]")
            
            return True
        else:
            print("数据预处理失败！")
            return False
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        print("\n" + "="*80)
        print("1. 描述性统计分析")
        print("="*80)
        
        # 主要变量的描述统计
        key_vars = ['年龄', '孕妇BMI', '孕周_数值', 'Y染色体浓度']
        desc_stats = self.male_data[key_vars].describe()
        
        print("主要变量描述统计:")
        print(desc_stats.round(4))
        
        # 添加偏度和峰度
        print("\n偏度和峰度分析:")
        for var in key_vars:
            skewness = stats.skew(self.male_data[var])
            kurtosis = stats.kurtosis(self.male_data[var])
            # Jarque-Bera正态性检验
            jb_stat, jb_pvalue = jarque_bera(self.male_data[var])
            print(f"{var}: 偏度={skewness:.3f}, 峰度={kurtosis:.3f}, JB检验p值={jb_pvalue:.4f}")
        
        # 分组统计
        print("\n分组统计分析:")
        
        # BMI分组统计
        bmi_groups = self.male_data.groupby('BMI分组')['Y染色体浓度'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        bmi_labels = {1: '正常(<25)', 2: '超重(25-28)', 3: '轻度肥胖(28-32)', 4: '中度肥胖(32-36)', 5: '重度肥胖(≥36)'}
        bmi_groups.index = [bmi_labels.get(i, f'组{i}') for i in bmi_groups.index]
        print("BMI分组下Y染色体浓度分布:")
        print(bmi_groups)
        
        # 孕期阶段统计
        stage_groups = self.male_data.groupby('孕期阶段')['Y染色体浓度'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        stage_labels = {1: '孕早期(<14周)', 2: '孕中期(14-21周)', 3: '孕晚期(≥21周)'}
        stage_groups.index = [stage_labels.get(i, f'阶段{i}') for i in stage_groups.index]
        print("\n孕期阶段下Y染色体浓度分布:")
        print(stage_groups)
        
        return desc_stats
    
    def correlation_analysis(self):
        """相关性分析"""
        print("\n" + "="*80)
        print("2. 相关性分析")
        print("="*80)
        
        # 主要连续变量
        continuous_vars = ['年龄', '孕妇BMI', '孕周_数值', 'Y染色体浓度']
        
        # Pearson相关性分析
        print("Pearson相关系数矩阵:")
        pearson_corr = self.male_data[continuous_vars].corr(method='pearson')
        print(pearson_corr.round(4))
        
        # Spearman秩相关分析
        print("\nSpearman秩相关系数矩阵:")
        spearman_corr = self.male_data[continuous_vars].corr(method='spearman')
        print(spearman_corr.round(4))
        
        # 详细的相关性检验
        print("\n详细相关性检验结果:")
        target_var = 'Y染色体浓度'
        independent_vars = ['年龄', '孕妇BMI', '孕周_数值']
        
        for var in independent_vars:
            # Pearson相关
            pearson_r, pearson_p = pearsonr(self.male_data[var], self.male_data[target_var])
            # Spearman相关
            spearman_r, spearman_p = spearmanr(self.male_data[var], self.male_data[target_var])
            
            print(f"\n{var} 与 {target_var}:")
            print(f"  Pearson: r={pearson_r:.4f}, p值={pearson_p:.6f}")
            print(f"  Spearman: ρ={spearman_r:.4f}, p值={spearman_p:.6f}")
            
            # 相关性强度解释
            abs_r = abs(pearson_r)
            if abs_r >= 0.7:
                strength = "强相关"
            elif abs_r >= 0.5:
                strength = "中等相关"
            elif abs_r >= 0.3:
                strength = "弱相关"
            else:
                strength = "很弱相关"
            
            direction = "正相关" if pearson_r > 0 else "负相关"
            significance = "显著" if pearson_p < 0.05 else "不显著"
            
            print(f"  结论: {direction}, {strength}, 统计学{significance}")
        
        # 保存相关性结果
        self.correlation_results = {
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }
        
        # 绘制相关性热力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('Pearson相关系数热力图', fontsize=14)
        
        sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('Spearman秩相关系数热力图', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pearson_corr, spearman_corr
    
    def group_difference_analysis(self):
        """分组差异分析"""
        print("\n" + "="*80)
        print("3. 分组差异分析")
        print("="*80)
        
        # BMI分组的Y染色体浓度差异分析
        print("BMI分组间Y染色体浓度差异检验:")
        
        bmi_groups_data = []
        bmi_group_names = []
        for group_id in sorted(self.male_data['BMI分组'].unique()):
            group_data = self.male_data[self.male_data['BMI分组'] == group_id]['Y染色体浓度']
            if len(group_data) > 0:
                bmi_groups_data.append(group_data)
                group_names = {1: '正常', 2: '超重', 3: '轻度肥胖', 4: '中度肥胖', 5: '重度肥胖'}
                bmi_group_names.append(group_names.get(group_id, f'组{group_id}'))
        
        if len(bmi_groups_data) > 2:
            # 方差齐性检验 (Levene检验)
            levene_stat, levene_p = stats.levene(*bmi_groups_data)
            print(f"Levene方差齐性检验: 统计量={levene_stat:.4f}, p值={levene_p:.6f}")
            
            # 单因素方差分析
            f_stat, f_p = f_oneway(*bmi_groups_data)
            print(f"单因素ANOVA: F={f_stat:.4f}, p值={f_p:.6f}")
            
            if f_p < 0.05:
                print("结论: BMI分组间Y染色体浓度存在显著差异")
                
                # 事后检验 (Tukey HSD)
                from scipy.stats import tukey_hsd
                tukey_result = tukey_hsd(*bmi_groups_data)
                print("\nTukey HSD事后检验结果:")
                for i in range(len(bmi_group_names)):
                    for j in range(i+1, len(bmi_group_names)):
                        print(f"{bmi_group_names[i]} vs {bmi_group_names[j]}: p值={tukey_result.pvalue[i,j]:.6f}")
            else:
                print("结论: BMI分组间Y染色体浓度无显著差异")
        
        # 孕期阶段分组差异分析
        print("\n孕期阶段间Y染色体浓度差异检验:")
        
        stage_groups_data = []
        stage_group_names = []
        for group_id in sorted(self.male_data['孕期阶段'].unique()):
            group_data = self.male_data[self.male_data['孕期阶段'] == group_id]['Y染色体浓度']
            if len(group_data) > 0:
                stage_groups_data.append(group_data)
                stage_names = {1: '孕早期', 2: '孕中期', 3: '孕晚期'}
                stage_group_names.append(stage_names.get(group_id, f'阶段{group_id}'))
        
        if len(stage_groups_data) > 2:
            # 方差齐性检验
            levene_stat, levene_p = stats.levene(*stage_groups_data)
            print(f"Levene方差齐性检验: 统计量={levene_stat:.4f}, p值={levene_p:.6f}")
            
            # 单因素方差分析
            f_stat, f_p = f_oneway(*stage_groups_data)
            print(f"单因素ANOVA: F={f_stat:.4f}, p值={f_p:.6f}")
            
            if f_p < 0.05:
                print("结论: 孕期阶段间Y染色体浓度存在显著差异")
            else:
                print("结论: 孕期阶段间Y染色体浓度无显著差异")
        
        # 可视化分组差异
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # BMI分组箱线图
        bmi_plot_data = []
        bmi_plot_labels = []
        for group_id in sorted(self.male_data['BMI分组'].unique()):
            group_data = self.male_data[self.male_data['BMI分组'] == group_id]['Y染色体浓度']
            if len(group_data) > 0:
                bmi_plot_data.append(group_data)
                group_names = {1: '正常', 2: '超重', 3: '轻度肥胖', 4: '中度肥胖', 5: '重度肥胖'}
                bmi_plot_labels.append(group_names.get(group_id, f'组{group_id}'))
        
        ax1.boxplot(bmi_plot_data, labels=bmi_plot_labels)
        ax1.set_title('BMI分组下Y染色体浓度分布', fontsize=14)
        ax1.set_ylabel('Y染色体浓度', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 孕期阶段箱线图
        stage_plot_data = []
        stage_plot_labels = []
        for group_id in sorted(self.male_data['孕期阶段'].unique()):
            group_data = self.male_data[self.male_data['孕期阶段'] == group_id]['Y染色体浓度']
            if len(group_data) > 0:
                stage_plot_data.append(group_data)
                stage_names = {1: '孕早期', 2: '孕中期', 3: '孕晚期'}
                stage_plot_labels.append(stage_names.get(group_id, f'阶段{group_id}'))
        
        ax2.boxplot(stage_plot_data, labels=stage_plot_labels)
        ax2.set_title('孕期阶段下Y染色体浓度分布', fontsize=14)
        ax2.set_ylabel('Y染色体浓度', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('group_difference_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def build_regression_models(self):
        """构建回归模型"""
        print("\n" + "="*80)
        print("4. 回归模型构建与检验")
        print("="*80)
        
        # 准备建模数据
        y = self.male_data['Y染色体浓度'].values
        
        # 模型1: 基础线性模型 (孕周 + BMI + 年龄)
        print("模型1: 基础线性回归模型")
        print("-" * 50)
        
        X1_vars = ['孕周_数值', '孕妇BMI', '年龄']
        X1 = self.male_data[X1_vars].values
        X1 = sm.add_constant(X1)  # 添加截距项
        
        model1 = sm.OLS(y, X1).fit()
        print(model1.summary())
        
        # 模型诊断
        print("\n模型1诊断:")
        self._model_diagnostics(model1, X1, y, "基础线性模型")
        
        # 模型2: 多项式回归模型 (添加孕周平方项)
        print("\n" + "="*60)
        print("模型2: 多项式回归模型")
        print("-" * 50)
        
        X2_vars = ['孕周_数值', '孕妇BMI', '年龄', '孕周平方']
        X2 = self.male_data[X2_vars].values
        X2 = sm.add_constant(X2)
        
        model2 = sm.OLS(y, X2).fit()
        print(model2.summary())
        
        print("\n模型2诊断:")
        self._model_diagnostics(model2, X2, y, "多项式回归模型")
        
        # 模型3: 交互作用模型
        print("\n" + "="*60)
        print("模型3: 交互作用模型")
        print("-" * 50)
        
        X3_vars = ['孕周_数值', '孕妇BMI', '年龄', '孕周BMI交互', 'BMI年龄交互']
        X3 = self.male_data[X3_vars].values
        X3 = sm.add_constant(X3)
        
        model3 = sm.OLS(y, X3).fit()
        print(model3.summary())
        
        print("\n模型3诊断:")
        self._model_diagnostics(model3, X3, y, "交互作用模型")
        
        # 模型比较
        print("\n" + "="*60)
        print("模型比较")
        print("-" * 50)
        
        models = {'基础模型': model1, '多项式模型': model2, '交互模型': model3}
        comparison_results = []
        
        for name, model in models.items():
            aic = model.aic
            bic = model.bic
            r2 = model.rsquared
            adj_r2 = model.rsquared_adj
            comparison_results.append({
                '模型': name,
                'R²': r2,
                '调整R²': adj_r2,
                'AIC': aic,
                'BIC': bic
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        print(comparison_df.round(4))
        
        # 保存模型结果
        self.model_results = {
            'basic_model': model1,
            'polynomial_model': model2,
            'interaction_model': model3,
            'comparison': comparison_df
        }
        
        # 选择最佳模型
        best_model_name = comparison_df.loc[comparison_df['调整R²'].idxmax(), '模型']
        best_model = models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name} (调整R² = {best_model.rsquared_adj:.4f})")
        
        return models
    
    def _model_diagnostics(self, model, X, y, model_name):
        """模型诊断"""
        # 残差分析
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # 正态性检验
        jb_stat, jb_pvalue = jarque_bera(residuals)
        print(f"残差正态性检验 (Jarque-Bera): 统计量={jb_stat:.4f}, p值={jb_pvalue:.6f}")
        
        # 异方差检验
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
        print(f"异方差检验 (Breusch-Pagan): 统计量={bp_stat:.4f}, p值={bp_pvalue:.6f}")
        
        # 自相关检验
        dw_stat = durbin_watson(residuals)
        print(f"自相关检验 (Durbin-Watson): 统计量={dw_stat:.4f}")
        
        # 多重共线性检验 (VIF)
        if X.shape[1] > 1:
            print("多重共线性检验 (VIF):")
            vif_data = []
            for i in range(1, X.shape[1]):  # 跳过常数项
                vif = variance_inflation_factor(X, i)
                vif_data.append(vif)
            
            var_names = ['孕周_数值', '孕妇BMI', '年龄']
            if X.shape[1] > 4:  # 如果有更多变量
                if 'polynomial' in model_name.lower():
                    var_names.append('孕周平方')
                elif 'interaction' in model_name.lower():
                    var_names.extend(['孕周BMI交互', 'BMI年龄交互'])
            
            for name, vif in zip(var_names[:len(vif_data)], vif_data):
                print(f"  {name}: VIF = {vif:.4f}")
                if vif > 10:
                    print(f"    警告: {name}存在严重多重共线性")
                elif vif > 5:
                    print(f"    注意: {name}存在中等程度多重共线性")
    
    def visualize_relationships(self):
        """可视化变量关系"""
        print("\n" + "="*80)
        print("5. 变量关系可视化")
        print("="*80)
        
        # 创建散点图矩阵
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Y染色体浓度 vs 孕周
        axes[0, 0].scatter(self.male_data['孕周_数值'], self.male_data['Y染色体浓度'], 
                          alpha=0.6, s=30)
        axes[0, 0].set_xlabel('孕周数')
        axes[0, 0].set_ylabel('Y染色体浓度')
        axes[0, 0].set_title('Y染色体浓度 vs 孕周数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加拟合线
        z = np.polyfit(self.male_data['孕周_数值'], self.male_data['Y染色体浓度'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.male_data['孕周_数值'], p(self.male_data['孕周_数值']), "r--", alpha=0.8)
        
        # Y染色体浓度 vs BMI
        axes[0, 1].scatter(self.male_data['孕妇BMI'], self.male_data['Y染色体浓度'], 
                          alpha=0.6, s=30)
        axes[0, 1].set_xlabel('孕妇BMI')
        axes[0, 1].set_ylabel('Y染色体浓度')
        axes[0, 1].set_title('Y染色体浓度 vs 孕妇BMI')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加拟合线
        z = np.polyfit(self.male_data['孕妇BMI'], self.male_data['Y染色体浓度'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.male_data['孕妇BMI'], p(self.male_data['孕妇BMI']), "r--", alpha=0.8)
        
        # Y染色体浓度 vs 年龄
        axes[0, 2].scatter(self.male_data['年龄'], self.male_data['Y染色体浓度'], 
                          alpha=0.6, s=30)
        axes[0, 2].set_xlabel('年龄')
        axes[0, 2].set_ylabel('Y染色体浓度')
        axes[0, 2].set_title('Y染色体浓度 vs 年龄')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 添加拟合线
        z = np.polyfit(self.male_data['年龄'], self.male_data['Y染色体浓度'], 1)
        p = np.poly1d(z)
        axes[0, 2].plot(self.male_data['年龄'], p(self.male_data['年龄']), "r--", alpha=0.8)
        
        # 孕周 vs BMI (着色显示Y浓度)
        scatter = axes[1, 0].scatter(self.male_data['孕周_数值'], self.male_data['孕妇BMI'], 
                                    c=self.male_data['Y染色体浓度'], cmap='viridis', 
                                    alpha=0.7, s=30)
        axes[1, 0].set_xlabel('孕周数')
        axes[1, 0].set_ylabel('孕妇BMI')
        axes[1, 0].set_title('孕周 vs BMI (颜色=Y染色体浓度)')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # 残差图 (使用最佳模型)
        if hasattr(self, 'model_results') and 'basic_model' in self.model_results:
            model = self.model_results['basic_model']
            residuals = model.resid
            fitted_values = model.fittedvalues
            
            axes[1, 1].scatter(fitted_values, residuals, alpha=0.6, s=30)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('拟合值')
            axes[1, 1].set_ylabel('残差')
            axes[1, 1].set_title('残差图')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Q-Q图
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 2])
            axes[1, 2].set_title('残差Q-Q图')
        
        plt.tight_layout()
        plt.savefig('variable_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_final_model(self):
        """生成最终的关系模型"""
        print("\n" + "="*80)
        print("6. 最终关系模型")
        print("="*80)
        
        if not hasattr(self, 'model_results'):
            print("请先运行回归分析！")
            return
        
        # 选择最佳模型
        best_model = self.model_results['basic_model']  # 可以根据比较结果调整
        
        print("基于统计分析的最终关系模型:")
        print("-" * 50)
        
        # 提取模型参数
        params = best_model.params
        pvalues = best_model.pvalues
        conf_int = best_model.conf_int()
        
        print("Y染色体浓度 = β₀ + β₁×孕周数 + β₂×孕妇BMI + β₃×年龄 + ε")
        print()
        print("参数估计结果:")
        
        var_names = ['截距', '孕周数', '孕妇BMI', '年龄']
        for i, (name, param, pval) in enumerate(zip(var_names, params, pvalues)):
            significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            conf_lower, conf_upper = conf_int[i]  # 修复：直接使用索引而不是iloc
            print(f"{name}: β{i} = {param:.6f}{significance}")
            print(f"  95%置信区间: [{conf_lower:.6f}, {conf_upper:.6f}]")
            print(f"  p值: {pval:.6f}")
            print()
        
        # 模型评价指标
        print("模型评价指标:")
        print(f"R² = {best_model.rsquared:.4f}")
        print(f"调整R² = {best_model.rsquared_adj:.4f}")
        print(f"F统计量 = {best_model.fvalue:.4f}")
        print(f"F检验p值 = {best_model.f_pvalue:.6f}")
        print(f"AIC = {best_model.aic:.4f}")
        print(f"BIC = {best_model.bic:.4f}")
        
        # 模型方程
        print("\n具体数值模型方程:")
        equation = f"Y染色体浓度 = {params[0]:.6f}"
        for i, (param, name) in enumerate(zip(params[1:], ['孕周数', '孕妇BMI', '年龄']), 1):
            sign = "+" if param >= 0 else ""
            equation += f" {sign}{param:.6f}×{name}"
        print(equation)
        
        # 实际预测示例
        print("\n模型预测示例:")
        example_cases = [
            {'孕周': 15, 'BMI': 22, '年龄': 28, '描述': '正常BMI，中等年龄，孕中期'},
            {'孕周': 20, 'BMI': 30, '年龄': 35, '描述': '高BMI，高龄，孕中期'},
            {'孕周': 12, 'BMI': 18, '年龄': 25, '描述': '低BMI，年轻，孕早期'},
        ]
        
        for case in example_cases:
            X_pred = np.array([[1, case['孕周'], case['BMI'], case['年龄']]])
            y_pred = best_model.predict(X_pred)[0]
            print(f"{case['描述']}: 预测Y染色体浓度 = {y_pred:.4f}")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("NIPT问题1：胎儿Y染色体浓度与孕妇指标相关特性分析")
        print("="*80)
        
        # 1. 数据加载和预处理
        if not self.load_and_preprocess_data():
            return False
        
        # 2. 描述性统计
        self.descriptive_statistics()
        
        # 3. 相关性分析
        self.correlation_analysis()
        
        # 4. 分组差异分析
        self.group_difference_analysis()
        
        # 5. 回归模型构建
        self.build_regression_models()
        
        # 6. 变量关系可视化
        self.visualize_relationships()
        
        # 7. 生成最终模型
        self.generate_final_model()
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        
        return True

# 主程序执行
if __name__ == "__main__":
    # 创建分析器
    analyzer = NIPTAnalyzer()
    
    # 运行完整分析
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n问题1分析报告已完成，包含以下内容：")
        print("1. 描述性统计分析")
        print("2. 相关性分析（Pearson和Spearman）")
        print("3. 分组差异分析（ANOVA检验）")
        print("4. 多种回归模型构建与比较")
        print("5. 模型诊断与显著性检验")
        print("6. 变量关系可视化")
        print("7. 最终关系模型确定")
        print("\n所有分析结果具有严格的统计学依据和显著性检验。")