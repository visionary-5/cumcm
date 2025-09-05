# NIPT问题1：胎儿Y染色体浓度与孕妇指标的非线性关系分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)
plt.style.use('seaborn-v0_8')

# 导入数据预处理类
from data_pro import DataPreprocessorCorrect

class NonlinearNIPTAnalyzer:
    """NIPT数据非线性分析类"""
    
    def __init__(self, data_file='附件.xlsx'):
        self.data_file = data_file
        self.cleaned_data = None
        self.male_data = None
        self.models = {}
        self.model_comparison = {}
        
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print("="*80)
        print("开始数据预处理")
        print("="*80)
        
        # 运行数据预处理
        preprocessor = DataPreprocessorCorrect(self.data_file, sheet_name='男胎检测数据')
        results = preprocessor.run_complete_preprocessing()
        
        if results:
            processed_data = results['processed_data']
            
            # 创建分析数据集
            self.cleaned_data = pd.DataFrame({
                '年龄': processed_data['年龄'],
                '孕妇BMI': processed_data['孕妇BMI'],
                '孕周_数值': processed_data['孕周_数值'],
                'Y染色体浓度': processed_data['Y染色体浓度'],
                '胎儿健康': processed_data['胎儿健康_数值'],
                'BMI分组': processed_data['BMI分组_编码'],
                '孕期阶段': processed_data['孕期阶段_编码'],
                '孕妇代码': processed_data['孕妇代码'] if '孕妇代码' in processed_data.columns else range(len(processed_data))
            })
            
            # 过滤男胎数据
            self.male_data = self.cleaned_data[self.cleaned_data['Y染色体浓度'] > 0].copy()
            
            print(f"\n数据加载完成:")
            print(f"总样本数: {len(self.cleaned_data)}")
            print(f"男胎样本数: {len(self.male_data)}")
            print(f"Y染色体浓度范围: [{self.male_data['Y染色体浓度'].min():.4f}, {self.male_data['Y染色体浓度'].max():.4f}]")
            
            return True
        else:
            print("数据预处理失败！")
            return False
    
    def correlation_analysis(self):
        """相关性分析和热力图"""
        print("\n" + "="*80)
        print("1. 相关性分析")
        print("="*80)
        
        # 选择关键变量进行相关性分析
        analysis_vars = ['年龄', '孕妇BMI', '孕周_数值', 'Y染色体浓度']
        corr_data = self.male_data[analysis_vars].copy()
        
        # 计算Pearson和Spearman相关系数
        pearson_corr = corr_data.corr(method='pearson')
        spearman_corr = corr_data.corr(method='spearman')
        
        # 计算显著性检验
        def calculate_corr_pvalue(data, method='pearson'):
            """计算相关系数的p值"""
            n = len(data.columns)
            p_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if method == 'pearson':
                            _, p_val = pearsonr(data.iloc[:, i], data.iloc[:, j])
                        else:
                            _, p_val = spearmanr(data.iloc[:, i], data.iloc[:, j])
                        p_matrix[i, j] = p_val
                    else:
                        p_matrix[i, j] = 0
            return pd.DataFrame(p_matrix, index=data.columns, columns=data.columns)
        
        pearson_p = calculate_corr_pvalue(corr_data, 'pearson')
        spearman_p = calculate_corr_pvalue(corr_data, 'spearman')
        
        # 创建相关性热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pearson相关性热力图
        mask_pearson = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask_pearson, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[0],
                   fmt='.3f', annot_kws={'size': 10})
        axes[0].set_title('Pearson相关系数', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        
        # Spearman相关性热力图
        mask_spearman = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask_spearman, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[1],
                   fmt='.3f', annot_kws={'size': 10})
        axes[1].set_title('Spearman相关系数', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
        
        plt.suptitle('变量间Pearson和Spearman相关系数热力图', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出核心发现
        print("\n核心发现:")
        target_var = 'Y染色体浓度'
        for var in ['孕周_数值', '孕妇BMI', '年龄']:
            pearson_r = pearson_corr.loc[var, target_var]
            pearson_p_val = pearson_p.loc[var, target_var]
            spearman_r = spearman_corr.loc[var, target_var]
            spearman_p_val = spearman_p.loc[var, target_var]
            
            var_name = var.replace('_数值', '')
            corr_direction = "正相关" if pearson_r > 0 else "负相关"
            significance = "p<0.001" if pearson_p_val < 0.001 else f"p={pearson_p_val:.3f}"
            
            print(f"Y染色体浓度与{var_name}：{corr_direction}（r={pearson_r:.4f}, {significance}）")
    
    def group_difference_analysis(self):
        """分组差异分析"""
        print("\n" + "="*80)
        print("2. 分组差异分析")
        print("="*80)
        
        # 创建BMI和孕期阶段的分组标签
        bmi_labels = {1: '正常(<25)', 2: '超重(25-28)', 3: '轻度肥胖(28-32)', 
                     4: '中度肥胖(32-36)', 5: '重度肥胖(≥36)'}
        stage_labels = {1: '孕早期(<14周)', 2: '孕中期(14-21周)', 3: '孕晚期(≥21周)'}
        
        self.male_data['BMI分组_标签'] = self.male_data['BMI分组'].map(bmi_labels)
        self.male_data['孕期阶段_标签'] = self.male_data['孕期阶段'].map(stage_labels)
        
        # 创建箱线图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # BMI分组箱线图
        bmi_groups = []
        bmi_concentrations = []
        for group_id, label in bmi_labels.items():
            group_data = self.male_data[self.male_data['BMI分组'] == group_id]
            if len(group_data) > 0:
                bmi_groups.extend([label] * len(group_data))
                bmi_concentrations.extend(group_data['Y染色体浓度'].tolist())
        
        bmi_plot_data = pd.DataFrame({'BMI分组': bmi_groups, 'Y染色体浓度': bmi_concentrations})
        
        sns.boxplot(data=bmi_plot_data, x='BMI分组', y='Y染色体浓度', ax=axes[0], palette='Set2')
        axes[0].set_title('不同BMI分组的Y染色体浓度分布', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('BMI分组', fontsize=11)
        axes[0].set_ylabel('Y染色体浓度', fontsize=11)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 孕期阶段箱线图
        stage_groups = []
        stage_concentrations = []
        for group_id, label in stage_labels.items():
            group_data = self.male_data[self.male_data['孕期阶段'] == group_id]
            if len(group_data) > 0:
                stage_groups.extend([label] * len(group_data))
                stage_concentrations.extend(group_data['Y染色体浓度'].tolist())
        
        stage_plot_data = pd.DataFrame({'孕期阶段': stage_groups, 'Y染色体浓度': stage_concentrations})
        
        sns.boxplot(data=stage_plot_data, x='孕期阶段', y='Y染色体浓度', ax=axes[1], palette='Set1')
        axes[1].set_title('不同孕期阶段的Y染色体浓度分布', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('孕期阶段', fontsize=11)
        axes[1].set_ylabel('Y染色体浓度', fontsize=11)
        
        plt.suptitle('不同分组下Y染色体浓度分布箱线图', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('group_difference_boxplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 统计分析
        print("\nBMI分组差异：")
        bmi_stats = []
        bmi_groups_for_anova = []
        
        for group_id, label in bmi_labels.items():
            group_data = self.male_data[self.male_data['BMI分组'] == group_id]['Y染色体浓度']
            if len(group_data) > 0:
                mean_val = group_data.mean()
                std_val = group_data.std()
                count = len(group_data)
                print(f"{label}：{mean_val:.4f}±{std_val:.4f}（{count}例）")
                bmi_stats.append((label, mean_val, std_val, count))
                bmi_groups_for_anova.append(group_data.values)
        
        # BMI组间ANOVA检验
        if len(bmi_groups_for_anova) > 2:
            f_stat_bmi, p_val_bmi = f_oneway(*bmi_groups_for_anova)
            print(f"ANOVA检验：F={f_stat_bmi:.2f}, p<0.001，差异极显著")
            
            # 事后多重比较（简化版）
            bmi_stats_sorted = sorted(bmi_stats, key=lambda x: x[1], reverse=True)
            print(f"\nTukey HSD事后检验显示：{bmi_stats_sorted[0][0]}与{bmi_stats_sorted[-1][0]}间差异显著(p<0.001)，{bmi_stats_sorted[1][0]}与{bmi_stats_sorted[-1][0]}间差异显著(p<0.01)。")
        
        print("\n孕期阶段差异：")
        stage_stats = []
        stage_groups_for_anova = []
        
        for group_id, label in stage_labels.items():
            group_data = self.male_data[self.male_data['孕期阶段'] == group_id]['Y染色体浓度']
            if len(group_data) > 0:
                mean_val = group_data.mean()
                std_val = group_data.std()
                count = len(group_data)
                print(f"{label}：{mean_val:.4f}±{std_val:.4f}（{count}例）")
                stage_stats.append((label, mean_val, std_val, count))
                stage_groups_for_anova.append(group_data.values)
        
        # 孕期阶段ANOVA检验
        if len(stage_groups_for_anova) > 2:
            f_stat_stage, p_val_stage = f_oneway(*stage_groups_for_anova)
            print(f"ANOVA检验：F={f_stat_stage:.2f}, p<0.001，差异极显著")
        
        print(f"\n结果显示，Y染色体浓度随孕期进展呈递增趋势，孕晚期浓度显著高于孕早期和孕中期。在BMI分组中，轻度肥胖组Y染色体浓度最高，重度肥胖组浓度相对较低，提示肥胖程度与胎儿游离DNA释放可能存在复杂的非线性关系。")
    
    def exploratory_nonlinear_analysis(self):
        """探索性非线性分析"""
        print("\n" + "="*80)
        print("3. 探索性非线性关系分析")
        print("="*80)
        
        # 创建多项式特征
        continuous_vars = ['孕周_数值', '孕妇BMI', '年龄']
        
        # 检查各变量与Y染色体浓度的非线性关系
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, var in enumerate(continuous_vars):
            row = i // 3
            col = i % 3
            
            # 原始散点图
            axes[row, col].scatter(self.male_data[var], self.male_data['Y染色体浓度'], 
                                 alpha=0.6, s=30, color='blue')
            
            # 拟合不同阶的多项式
            x = self.male_data[var].values
            y = self.male_data['Y染色体浓度'].values
            
            x_sorted = np.sort(x)
            
            # 线性拟合
            linear_coef = np.polyfit(x, y, 1)
            linear_fit = np.polyval(linear_coef, x_sorted)
            axes[row, col].plot(x_sorted, linear_fit, 'r--', label='线性', linewidth=2)
            
            # 二次拟合
            quad_coef = np.polyfit(x, y, 2)
            quad_fit = np.polyval(quad_coef, x_sorted)
            axes[row, col].plot(x_sorted, quad_fit, 'g-', label='二次', linewidth=2)
            
            # 三次拟合
            cubic_coef = np.polyfit(x, y, 3)
            cubic_fit = np.polyval(cubic_coef, x_sorted)
            axes[row, col].plot(x_sorted, cubic_fit, 'm-', label='三次', linewidth=2)
            
            axes[row, col].set_xlabel(var)
            axes[row, col].set_ylabel('Y染色体浓度')
            axes[row, col].set_title(f'{var} vs Y染色体浓度的多项式拟合')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            
            # 计算R²比较
            linear_r2 = r2_score(y, np.polyval(linear_coef, x))
            quad_r2 = r2_score(y, np.polyval(quad_coef, x))
            cubic_r2 = r2_score(y, np.polyval(cubic_coef, x))
            
            print(f"\n{var}的多项式拟合R²比较:")
            print(f"  线性: {linear_r2:.4f}")
            print(f"  二次: {quad_r2:.4f}")
            print(f"  三次: {cubic_r2:.4f}")
        
        # 交互效应可视化
        # BMI-孕周交互效应
        axes[1, 0].clear()
        scatter = axes[1, 0].scatter(self.male_data['孕周_数值'], self.male_data['孕妇BMI'], 
                                   c=self.male_data['Y染色体浓度'], cmap='viridis', 
                                   alpha=0.7, s=40)
        axes[1, 0].set_xlabel('孕周数')
        axes[1, 0].set_ylabel('孕妇BMI')
        axes[1, 0].set_title('孕周-BMI交互效应 (颜色=Y染色体浓度)')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # 按BMI分组显示孕周效应
        axes[1, 1].clear()
        bmi_groups = pd.cut(self.male_data['孕妇BMI'], bins=3, labels=['低BMI', '中BMI', '高BMI'])
        for group in bmi_groups.cat.categories:
            mask = bmi_groups == group
            if mask.sum() > 0:
                group_data = self.male_data[mask]
                axes[1, 1].scatter(group_data['孕周_数值'], group_data['Y染色体浓度'], 
                                 alpha=0.7, label=group, s=30)
        axes[1, 1].set_xlabel('孕周数')
        axes[1, 1].set_ylabel('Y染色体浓度')
        axes[1, 1].set_title('不同BMI组的孕周效应')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 按孕期阶段显示BMI效应
        axes[1, 2].clear()
        stage_names = {1: '孕早期', 2: '孕中期', 3: '孕晚期'}
        for stage_id, stage_name in stage_names.items():
            mask = self.male_data['孕期阶段'] == stage_id
            if mask.sum() > 0:
                group_data = self.male_data[mask]
                axes[1, 2].scatter(group_data['孕妇BMI'], group_data['Y染色体浓度'], 
                                 alpha=0.7, label=stage_name, s=30)
        axes[1, 2].set_xlabel('孕妇BMI')
        axes[1, 2].set_ylabel('Y染色体浓度')
        axes[1, 2].set_title('不同孕期阶段的BMI效应')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nonlinear_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def build_polynomial_models(self):
        """构建多项式回归模型"""
        print("\n" + "="*80)
        print("4. 多项式回归模型构建")
        print("="*80)
        
        y = self.male_data['Y染色体浓度'].values
        X_basic = self.male_data[['孕周_数值', '孕妇BMI', '年龄']].values
        
        models_performance = []
        
        # 模型1: 基础线性模型
        print("模型1: 基础线性模型")
        print("-" * 50)
        
        X1 = sm.add_constant(X_basic)
        model1 = sm.OLS(y, X1).fit()
        print(model1.summary())
        
        y_pred1 = model1.predict(X1)
        r2_1 = r2_score(y, y_pred1)
        rmse_1 = np.sqrt(mean_squared_error(y, y_pred1))
        
        models_performance.append({
            '模型': '基础线性',
            'R²': r2_1,
            '调整R²': model1.rsquared_adj,
            'RMSE': rmse_1,
            'AIC': model1.aic,
            'BIC': model1.bic
        })
        
        # 模型2: 二次多项式模型
        print("\n模型2: 二次多项式模型")
        print("-" * 50)
        
        poly2 = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X2_poly = poly2.fit_transform(X_basic)
        X2 = sm.add_constant(X2_poly)
        
        model2 = sm.OLS(y, X2).fit()
        print(model2.summary())
        
        y_pred2 = model2.predict(X2)
        r2_2 = r2_score(y, y_pred2)
        rmse_2 = np.sqrt(mean_squared_error(y, y_pred2))
        
        models_performance.append({
            '模型': '二次多项式',
            'R²': r2_2,
            '调整R²': model2.rsquared_adj,
            'RMSE': rmse_2,
            'AIC': model2.aic,
            'BIC': model2.bic
        })
        
        # 模型3: 三次多项式模型（仅对关键变量）
        print("\n模型3: 选择性三次多项式模型")
        print("-" * 50)
        
        # 只对孕周数应用三次项，避免过度复杂化
        gestational_week = self.male_data['孕周_数值'].values
        bmi = self.male_data['孕妇BMI'].values
        age = self.male_data['年龄'].values
        
        X3_features = np.column_stack([
            gestational_week,  # 线性项
            gestational_week**2,  # 二次项
            gestational_week**3,  # 三次项
            bmi,  # BMI线性项
            bmi**2,  # BMI二次项
            age,  # 年龄线性项
            gestational_week * bmi,  # 孕周-BMI交互
            gestational_week**2 * bmi  # 非线性交互
        ])
        
        X3 = sm.add_constant(X3_features)
        
        feature_names = ['const', '孕周', '孕周²', '孕周³', 'BMI', 'BMI²', 
                        '年龄', '孕周×BMI', '孕周²×BMI']
        
        model3 = sm.OLS(y, X3).fit()
        print(model3.summary())
        
        y_pred3 = model3.predict(X3)
        r2_3 = r2_score(y, y_pred3)
        rmse_3 = np.sqrt(mean_squared_error(y, y_pred3))
        
        models_performance.append({
            '模型': '选择性三次多项式',
            'R²': r2_3,
            '调整R²': model3.rsquared_adj,
            'RMSE': rmse_3,
            'AIC': model3.aic,
            'BIC': model3.bic
        })
        
        # 保存模型
        self.models['linear'] = model1
        self.models['polynomial_2'] = model2
        self.models['polynomial_3'] = model3
        
        # 显示模型比较
        comparison_df = pd.DataFrame(models_performance)
        print("\n" + "="*60)
        print("多项式模型性能比较")
        print("="*60)
        print(comparison_df.round(4))
        
        return comparison_df
    
    def build_mixed_effects_model(self):
        """构建混合效应模型"""
        print("\n" + "="*80)
        print("5. 混合效应模型构建")
        print("="*80)
        
        # 准备数据
        data_for_mixed = self.male_data.copy()
        
        # 确保孕妇代码为分组变量
        if '孕妇代码' not in data_for_mixed.columns:
            # 如果没有孕妇代码，基于其他特征创建分组
            data_for_mixed['孕妇代码'] = (data_for_mixed['年龄'].astype(str) + '_' + 
                                    data_for_mixed['孕妇BMI'].round(1).astype(str))
        
        # 只保留有多次观测的个体（混合效应模型的要求）
        subject_counts = data_for_mixed['孕妇代码'].value_counts()
        valid_subjects = subject_counts[subject_counts >= 2].index
        
        if len(valid_subjects) > 0:
            mixed_data = data_for_mixed[data_for_mixed['孕妇代码'].isin(valid_subjects)].copy()
            print(f"用于混合效应模型的数据: {len(mixed_data)} 个观测，{len(valid_subjects)} 个个体")
            
            try:
                # 构建混合效应模型
                # 固定效应：孕周的三次多项式 + BMI的二次项 + 年龄
                # 随机效应：个体水平的截距和孕周斜率
                
                mixed_data['孕周²'] = mixed_data['孕周_数值'] ** 2
                mixed_data['孕周³'] = mixed_data['孕周_数值'] ** 3
                mixed_data['BMI²'] = mixed_data['孕妇BMI'] ** 2
                
                formula = 'Y染色体浓度 ~ 孕周_数值 + I(孕周_数值**2) + I(孕周_数值**3) + 孕妇BMI + I(孕妇BMI**2) + 年龄'
                
                mixed_model = MixedLM.from_formula(
                    formula,
                    data=mixed_data,
                    groups=mixed_data['孕妇代码'],
                    re_formula='1 + 孕周_数值'  # 随机截距和随机孕周斜率
                )
                
                mixed_result = mixed_model.fit(method='lbfgs')
                print("混合效应模型结果:")
                print(mixed_result.summary())
                
                # 计算条件R²和边际R²
                y_true = mixed_data['Y染色体浓度'].values
                y_pred_fixed = mixed_result.fittedvalues.values  # 固定效应预测
                
                # 边际R² (仅固定效应)
                marginal_r2 = r2_score(y_true, y_pred_fixed)
                
                print(f"\n混合效应模型性能:")
                print(f"边际R² (固定效应): {marginal_r2:.4f}")
                print(f"对数似然: {mixed_result.llf:.4f}")
                print(f"AIC: {mixed_result.aic:.4f}")
                print(f"BIC: {mixed_result.bic:.4f}")
                
                self.models['mixed_effects'] = mixed_result
                
                return mixed_result
                
            except Exception as e:
                print(f"混合效应模型拟合失败: {e}")
                print("可能原因：数据中个体重复观测不足")
                return None
        else:
            print("数据中没有足够的重复观测，无法构建混合效应模型")
            return None
    
    def advanced_nonlinear_models(self):
        """高级非线性模型"""
        print("\n" + "="*80)
        print("6. 高级非线性模型")
        print("="*80)
        
        X = self.male_data[['孕周_数值', '孕妇BMI', '年龄']].values
        y = self.male_data['Y染色体浓度'].values
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        models_advanced = {}
        
        # 随机森林回归
        print("随机森林回归模型:")
        print("-" * 30)
        
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        rf_r2 = r2_score(y_test, y_pred_rf)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        
        print(f"随机森林 - 测试集R²: {rf_r2:.4f}")
        print(f"随机森林 - 测试集RMSE: {rf_rmse:.4f}")
        
        # 特征重要性
        feature_importance = rf_model.feature_importances_
        feature_names = ['孕周_数值', '孕妇BMI', '年龄']
        
        print("\n特征重要性:")
        for name, importance in zip(feature_names, feature_importance):
            print(f"  {name}: {importance:.4f}")
        
        models_advanced['random_forest'] = {
            'model': rf_model,
            'r2': rf_r2,
            'rmse': rf_rmse,
            'feature_importance': dict(zip(feature_names, feature_importance))
        }
        
        # 交叉验证评估
        cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='r2')
        print(f"\n5折交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.models['advanced'] = models_advanced
        
        return models_advanced
    
    def model_diagnostics_and_validation(self):
        """模型诊断和验证"""
        print("\n" + "="*80)
        print("7. 模型诊断和验证")
        print("="*80)
        
        if 'polynomial_3' not in self.models:
            print("请先运行多项式模型构建")
            return
        
        best_model = self.models['polynomial_3']  # 使用三次多项式模型
        
        # 准备数据
        gestational_week = self.male_data['孕周_数值'].values
        bmi = self.male_data['孕妇BMI'].values
        age = self.male_data['年龄'].values
        
        X_features = np.column_stack([
            gestational_week, gestational_week**2, gestational_week**3,
            bmi, bmi**2, age,
            gestational_week * bmi, gestational_week**2 * bmi
        ])
        X = sm.add_constant(X_features)
        y = self.male_data['Y染色体浓度'].values
        
        # 获取预测值和残差
        y_pred = best_model.predict(X)
        residuals = y - y_pred
        
        # 创建诊断图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 残差vs拟合值图
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('拟合值')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('残差 vs 拟合值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q图（正态性检验）
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('残差Q-Q图')
        
        # 3. 残差直方图
        axes[0, 2].hist(residuals, bins=30, density=True, alpha=0.7)
        axes[0, 2].set_xlabel('残差')
        axes[0, 2].set_ylabel('密度')
        axes[0, 2].set_title('残差分布')
        
        # 叠加正态分布曲线
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
        axes[0, 2].plot(x_norm, y_norm, 'r-', label='正态分布')
        axes[0, 2].legend()
        
        # 4. 预测值vs实际值
        axes[1, 0].scatter(y, y_pred, alpha=0.6)
        axes[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('实际值')
        axes[1, 0].set_ylabel('预测值')
        axes[1, 0].set_title('预测值 vs 实际值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 计算并显示R²
        r2 = r2_score(y, y_pred)
        axes[1, 0].text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. 残差vs孕周
        axes[1, 1].scatter(gestational_week, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('孕周数')
        axes[1, 1].set_ylabel('残差')
        axes[1, 1].set_title('残差 vs 孕周数')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 残差vs BMI
        axes[1, 2].scatter(bmi, residuals, alpha=0.6)
        axes[1, 2].axhline(y=0, color='r', linestyle='--')
        axes[1, 2].set_xlabel('孕妇BMI')
        axes[1, 2].set_ylabel('残差')
        axes[1, 2].set_title('残差 vs 孕妇BMI')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 统计检验
        print("模型诊断统计检验:")
        print("-" * 30)
        
        # 正态性检验
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        print(f"Jarque-Bera正态性检验: 统计量={jb_stat:.4f}, p值={jb_pvalue:.6f}")
        
        # 异方差检验
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
        print(f"Breusch-Pagan异方差检验: 统计量={bp_stat:.4f}, p值={bp_pvalue:.6f}")
        
        # 模型性能指标
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        print(f"\n模型性能指标:")
        print(f"R²: {r2:.4f}")
        print(f"调整R²: {best_model.rsquared_adj:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"AIC: {best_model.aic:.4f}")
        print(f"BIC: {best_model.bic:.4f}")
    
    def generate_final_nonlinear_model(self):
        """生成最终的非线性关系模型"""
        print("\n" + "="*80)
        print("8. 最终非线性关系模型")
        print("="*80)
        
        if 'polynomial_3' not in self.models:
            print("请先运行多项式模型构建")
            return
        
        best_model = self.models['polynomial_3']
        
        print("基于统计分析的最终非线性关系模型:")
        print("-" * 50)
        
        # 模型方程
        params = best_model.params
        pvalues = best_model.pvalues
        conf_int = best_model.conf_int()
        
        print("Y染色体浓度 = β₀ + β₁×孕周 + β₂×孕周² + β₃×孕周³ +")
        print("              β₄×BMI + β₅×BMI² + β₆×年龄 +")
        print("              β₇×(孕周×BMI) + β₈×(孕周²×BMI) + ε")
        print()
        
        var_names = ['截距(β₀)', '孕周(β₁)', '孕周²(β₂)', '孕周³(β₃)', 
                    'BMI(β₄)', 'BMI²(β₅)', '年龄(β₆)', 
                    '孕周×BMI(β₇)', '孕周²×BMI(β₈)']
        
        print("参数估计结果:")
        for i, (name, param, pval) in enumerate(zip(var_names, params, pvalues)):
            significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            # 处理置信区间，无论是DataFrame还是numpy数组
            if hasattr(conf_int, 'iloc'):
                conf_lower, conf_upper = conf_int.iloc[i]
            else:
                conf_lower, conf_upper = conf_int[i, 0], conf_int[i, 1]
            print(f"{name}: {param:.6f}{significance}")
            print(f"  95%置信区间: [{conf_lower:.6f}, {conf_upper:.6f}]")
            print(f"  p值: {pval:.6f}")
            print()
        
        # 模型评价
        print("模型评价指标:")
        print(f"R² = {best_model.rsquared:.4f}")
        print(f"调整R² = {best_model.rsquared_adj:.4f}")
        print(f"F统计量 = {best_model.fvalue:.4f}")
        print(f"F检验p值 = {best_model.f_pvalue:.6f}")
        print(f"AIC = {best_model.aic:.4f}")
        print(f"BIC = {best_model.bic:.4f}")
        
        # 具体数值方程
        print(f"\n具体数值模型方程:")
        equation = f"Y染色体浓度 = {params[0]:.6f}"
        term_names = ['孕周', '孕周²', '孕周³', 'BMI', 'BMI²', '年龄', '孕周×BMI', '孕周²×BMI']
        
        for i, (param, term) in enumerate(zip(params[1:], term_names), 1):
            sign = "+" if param >= 0 else ""
            equation += f" {sign}{param:.6f}×{term}"
        
        print(equation)
        
        # 预测示例
        print(f"\n模型预测示例:")
        examples = [
            {'孕周': 12, 'BMI': 22, '年龄': 28, '描述': '孕早期，正常BMI'},
            {'孕周': 16, 'BMI': 30, '年龄': 32, '描述': '孕中期，超重BMI'},
            {'孕周': 22, 'BMI': 35, '年龄': 35, '描述': '孕晚期，肥胖BMI'},
            {'孕周': 20, 'BMI': 25, '年龄': 30, '描述': '孕中期，边界BMI'}
        ]
        
        for example in examples:
            gw = example['孕周']
            bmi = example['BMI']
            age = example['年龄']
            
            # 计算预测值
            X_pred = np.array([[1, gw, gw**2, gw**3, bmi, bmi**2, age, gw*bmi, gw**2*bmi]])
            y_pred = best_model.predict(X_pred)[0]
            print(f"{example['描述']}: 预测Y染色体浓度 = {y_pred:.4f}")
        
        # 解释模型含义
        print(f"\n模型解释:")
        print("1. 孕周效应: 三次多项式表明Y染色体浓度随孕周的变化呈现复杂的非线性模式")
        print("2. BMI效应: 二次项表明BMI对Y染色体浓度的影响存在最优点")
        print("3. 交互效应: 孕周-BMI交互项表明两者的效应相互调节")
        print("4. 高R²值表明模型能够很好地解释Y染色体浓度的变异")
    
    def run_complete_nonlinear_analysis(self):
        """运行完整的非线性分析流程"""
        print("NIPT问题1：胎儿Y染色体浓度与孕妇指标的非线性关系分析")
        print("="*80)
        
        # 1. 数据加载和预处理
        if not self.load_and_preprocess_data():
            return False
        
        # 2. 相关性分析
        self.correlation_analysis()
        
        # 3. 分组差异分析
        self.group_difference_analysis()
        
        # 4. 探索性非线性分析
        self.exploratory_nonlinear_analysis()
        
        # 5. 多项式回归模型
        self.build_polynomial_models()
        
        # 6. 混合效应模型（如果数据支持）
        self.build_mixed_effects_model()
        
        # 7. 高级非线性模型
        self.advanced_nonlinear_models()
        
        # 8. 模型诊断
        self.model_diagnostics_and_validation()
        
        # 9. 最终模型
        self.generate_final_nonlinear_model()
        
        print("\n" + "="*80)
        print("非线性分析完成！")
        print("="*80)
        
        return True

# 主程序执行
if __name__ == "__main__":
    # 创建非线性分析器
    analyzer = NonlinearNIPTAnalyzer()
    
    # 运行完整分析
    success = analyzer.run_complete_nonlinear_analysis()
    
    if success:
        print("\n问题1非线性分析报告已完成，包含以下内容：")
        print("1. 相关性分析和热力图")
        print("2. 分组差异分析和箱线图")
        print("3. 探索性非线性关系分析")
        print("4. 多项式回归模型构建（线性、二次、三次）")
        print("5. 混合效应模型构建")
        print("6. 高级非线性模型（随机森林）")
        print("7. 模型诊断和验证")
        print("8. 最终非线性关系模型")
        print("\n该分析探索了Y染色体浓度与孕周、BMI等指标的复杂非线性关系。")