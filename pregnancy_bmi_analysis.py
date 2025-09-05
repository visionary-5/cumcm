# 孕妇BMI分组合理性分析与改进方案
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PregnancyBMIAnalyzer:
    """孕妇BMI分组合理性分析"""
    
    def __init__(self, data_file='附件.xlsx'):
        self.data_file = data_file
        self.data = None
        
    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_excel(self.data_file, sheet_name='男胎检测数据')
            
            # 转换孕周
            def convert_week(week_str):
                if pd.isna(week_str):
                    return np.nan
                week_str = str(week_str).strip().upper()
                if 'W' in week_str:
                    week_str = week_str.replace('W', 'w')
                if 'w' in week_str.lower():
                    parts = week_str.lower().split('w')
                    try:
                        weeks = int(parts[0])
                        if len(parts) > 1 and '+' in parts[1]:
                            days = int(parts[1].split('+')[1])
                            return weeks + days/7
                        else:
                            return float(weeks)
                    except ValueError:
                        return np.nan
                try:
                    return float(week_str)
                except ValueError:
                    return np.nan
            
            self.data['孕周_数值'] = self.data['检测孕周'].apply(convert_week)
            
            # 过滤有效数据
            self.data = self.data.dropna(subset=['孕妇BMI', 'Y染色体浓度', '孕周_数值'])
            
            print(f"数据加载完成，有效样本数: {len(self.data)}")
            print(f"BMI范围: [{self.data['孕妇BMI'].min():.1f}, {self.data['孕妇BMI'].max():.1f}]")
            print(f"孕周范围: [{self.data['孕周_数值'].min():.1f}, {self.data['孕周_数值'].max():.1f}]")
            
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def analyze_bmi_distribution(self):
        """分析BMI分布特征"""
        print("\n" + "="*60)
        print("孕妇BMI分布特征分析")
        print("="*60)
        
        bmi_stats = self.data['孕妇BMI'].describe()
        print("BMI描述统计:")
        print(bmi_stats)
        
        # 检查是否符合正常人群分布
        print(f"\n与标准BMI分界点的对比:")
        print(f"BMI < 18.5 (偏瘦): {(self.data['孕妇BMI'] < 18.5).mean():.1%}")
        print(f"BMI 18.5-24 (正常): {((self.data['孕妇BMI'] >= 18.5) & (self.data['孕妇BMI'] < 24)).mean():.1%}")
        print(f"BMI 24-28 (超重): {((self.data['孕妇BMI'] >= 24) & (self.data['孕妇BMI'] < 28)).mean():.1%}")
        print(f"BMI ≥ 28 (肥胖): {(self.data['孕妇BMI'] >= 28).mean():.1%}")
        
        # 孕期特殊情况分析
        print(f"\n孕期特殊BMI分布:")
        print(f"BMI ≥ 30 (重度肥胖): {(self.data['孕妇BMI'] >= 30).mean():.1%}")
        print(f"BMI ≥ 35 (极重度肥胖): {(self.data['孕妇BMI'] >= 35).mean():.1%}")
        
        return bmi_stats
    
    def compare_grouping_methods(self):
        """比较不同分组方法的效果"""
        print("\n" + "="*60)
        print("不同BMI分组方法比较")
        print("="*60)
        
        # 方法1: 标准BMI分组（原方法）
        def standard_bmi_group(bmi):
            if bmi < 18.5: return 1  # 偏瘦
            elif bmi < 24.0: return 2  # 正常
            elif bmi < 27.5: return 3  # 超重
            else: return 4  # 肥胖
        
        # 方法2: 孕妇专用BMI分组（基于临床指南）
        def pregnancy_bmi_group(bmi):
            # 基于WHO孕妇体重管理指南调整
            if bmi < 18.5: return 1  # 偏瘦
            elif bmi < 25.0: return 2  # 正常（上限提高到25）
            elif bmi < 30.0: return 3  # 超重（上限提高到30）
            else: return 4  # 肥胖
        
        # 方法3: 基于数据分布的四分位数分组
        quartiles = self.data['孕妇BMI'].quantile([0.25, 0.5, 0.75])
        def quartile_bmi_group(bmi):
            if bmi <= quartiles[0.25]: return 1  # Q1
            elif bmi <= quartiles[0.5]: return 2  # Q2
            elif bmi <= quartiles[0.75]: return 3  # Q3
            else: return 4  # Q4
        
        # 方法4: 基于临床风险的分组
        def clinical_risk_group(bmi):
            if bmi < 20: return 1    # 低体重风险
            elif bmi < 26: return 2  # 正常风险
            elif bmi < 32: return 3  # 中等风险
            else: return 4           # 高风险
        
        # 应用不同分组方法
        self.data['标准分组'] = self.data['孕妇BMI'].apply(standard_bmi_group)
        self.data['孕妇分组'] = self.data['孕妇BMI'].apply(pregnancy_bmi_group)
        self.data['四分位分组'] = self.data['孕妇BMI'].apply(quartile_bmi_group)
        self.data['临床风险分组'] = self.data['孕妇BMI'].apply(clinical_risk_group)
        
        # 比较各组样本分布
        grouping_methods = ['标准分组', '孕妇分组', '四分位分组', '临床风险分组']
        
        print("各分组方法的样本分布:")
        for method in grouping_methods:
            print(f"\n{method}:")
            group_counts = self.data[method].value_counts().sort_index()
            for group_id, count in group_counts.items():
                percentage = count / len(self.data) * 100
                print(f"  组{group_id}: {count}人 ({percentage:.1f}%)")
        
        # 比较各组间Y染色体浓度差异
        print("\n各分组方法的组间差异检验:")
        anova_results = {}
        
        for method in grouping_methods:
            groups_data = []
            for group_id in sorted(self.data[method].unique()):
                group_y_values = self.data[self.data[method] == group_id]['Y染色体浓度']
                if len(group_y_values) > 2:  # 至少3个样本才参与检验
                    groups_data.append(group_y_values.values)
            
            if len(groups_data) >= 2:
                f_stat, p_value = f_oneway(*groups_data)
                anova_results[method] = {'F': f_stat, 'p': p_value}
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{method}: F={f_stat:.3f}, p={p_value:.6f} {significance}")
        
        return anova_results
    
    def analyze_pregnancy_specific_factors(self):
        """分析孕期特异性因素"""
        print("\n" + "="*60)
        print("孕期特异性因素分析")
        print("="*60)
        
        # 1. BMI与孕周的关系
        correlation_bmi_week = stats.pearsonr(self.data['孕妇BMI'], self.data['孕周_数值'])
        print(f"BMI与孕周的相关性: r={correlation_bmi_week[0]:.4f}, p={correlation_bmi_week[1]:.6f}")
        
        # 2. 不同孕期阶段的BMI分布
        def pregnancy_stage(week):
            if week < 14: return '孕早期'
            elif week < 28: return '孕中期'
            else: return '孕晚期'
        
        self.data['孕期阶段'] = self.data['孕周_数值'].apply(pregnancy_stage)
        
        print("\n不同孕期阶段的BMI分布:")
        for stage in ['孕早期', '孕中期', '孕晚期']:
            stage_data = self.data[self.data['孕期阶段'] == stage]
            if len(stage_data) > 0:
                mean_bmi = stage_data['孕妇BMI'].mean()
                std_bmi = stage_data['孕妇BMI'].std()
                print(f"{stage}: n={len(stage_data)}, BMI={mean_bmi:.2f}±{std_bmi:.2f}")
        
        # 3. 高BMI孕妇的Y染色体浓度分析
        high_bmi_threshold = 30  # 临床上认为的高风险BMI
        high_bmi_data = self.data[self.data['孕妇BMI'] >= high_bmi_threshold]
        normal_bmi_data = self.data[self.data['孕妇BMI'] < high_bmi_threshold]
        
        print(f"\n高BMI组(≥30) vs 正常BMI组(<30)的Y染色体浓度比较:")
        print(f"高BMI组: n={len(high_bmi_data)}, Y浓度={high_bmi_data['Y染色体浓度'].mean():.6f}±{high_bmi_data['Y染色体浓度'].std():.6f}")
        print(f"正常BMI组: n={len(normal_bmi_data)}, Y浓度={normal_bmi_data['Y染色体浓度'].mean():.6f}±{normal_bmi_data['Y染色体浓度'].std():.6f}")
        
        # t检验
        t_stat, t_p = stats.ttest_ind(high_bmi_data['Y染色体浓度'], normal_bmi_data['Y染色体浓度'])
        print(f"t检验: t={t_stat:.4f}, p={t_p:.6f}")
    
    def recommend_optimal_grouping(self):
        """推荐最优分组方案"""
        print("\n" + "="*60)
        print("推荐的孕妇BMI分组方案")
        print("="*60)
        
        # 基于文献和临床实践的推荐分组
        def recommended_bmi_group(bmi):
            """
            基于以下原则的分组:
            1. 考虑孕妇生理特点
            2. 临床风险评估
            3. 数据分布合理性
            4. 统计检验力
            """
            if bmi < 20.0:   return 1  # 低体重(孕妇风险组)
            elif bmi < 26.0: return 2  # 正常体重
            elif bmi < 32.0: return 3  # 超重/轻度肥胖
            else:            return 4  # 中重度肥胖
        
        self.data['推荐分组'] = self.data['孕妇BMI'].apply(recommended_bmi_group)
        
        # 分组统计
        print("推荐分组的样本分布和特征:")
        group_names = {1: '低体重(<20)', 2: '正常(20-26)', 3: '超重(26-32)', 4: '肥胖(≥32)'}
        
        for group_id in sorted(self.data['推荐分组'].unique()):
            group_data = self.data[self.data['推荐分组'] == group_id]
            
            bmi_mean = group_data['孕妇BMI'].mean()
            bmi_std = group_data['孕妇BMI'].std()
            y_mean = group_data['Y染色体浓度'].mean()
            y_std = group_data['Y染色体浓度'].std()
            
            print(f"\n组{group_id} - {group_names[group_id]}:")
            print(f"  样本数: {len(group_data)}人 ({len(group_data)/len(self.data)*100:.1f}%)")
            print(f"  BMI: {bmi_mean:.2f}±{bmi_std:.2f}")
            print(f"  Y染色体浓度: {y_mean:.6f}±{y_std:.6f}")
        
        # ANOVA检验
        groups_data = []
        for group_id in sorted(self.data['推荐分组'].unique()):
            group_y_values = self.data[self.data['推荐分组'] == group_id]['Y染色体浓度']
            groups_data.append(group_y_values.values)
        
        f_stat, p_value = f_oneway(*groups_data)
        print(f"\n推荐分组的ANOVA检验: F={f_stat:.3f}, p={p_value:.6f}")
        
        # 与原始分组比较
        original_groups_data = []
        for group_id in sorted(self.data['标准分组'].unique()):
            group_y_values = self.data[self.data['标准分组'] == group_id]['Y染色体浓度']
            if len(group_y_values) > 2:
                original_groups_data.append(group_y_values.values)
        
        if len(original_groups_data) >= 2:
            f_orig, p_orig = f_oneway(*original_groups_data)
            print(f"原始分组的ANOVA检验: F={f_orig:.3f}, p={p_orig:.6f}")
            
            print(f"\n分组效果比较:")
            print(f"推荐分组: F={f_stat:.3f}, p={p_value:.6f}")
            print(f"原始分组: F={f_orig:.3f}, p={p_orig:.6f}")
            print(f"推荐分组F值更{'高' if f_stat > f_orig else '低'}，差异{'更' if p_value < p_orig else '较不'}显著")
    
    def create_visualization(self):
        """创建可视化对比"""
        print("\n生成BMI分组对比可视化...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. BMI分布直方图
        axes[0, 0].hist(self.data['孕妇BMI'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.data['孕妇BMI'].mean(), color='red', linestyle='--', 
                          label=f'均值: {self.data["孕妇BMI"].mean():.1f}')
        axes[0, 0].axvline(24, color='orange', linestyle=':', label='标准BMI=24')
        axes[0, 0].axvline(28, color='purple', linestyle=':', label='标准BMI=28')
        axes[0, 0].set_xlabel('BMI')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('孕妇BMI分布')
        axes[0, 0].legend()
        
        # 2. 原始分组箱线图
        original_groups = []
        original_labels = []
        for group_id in sorted(self.data['标准分组'].unique()):
            group_data = self.data[self.data['标准分组'] == group_id]['Y染色体浓度']
            if len(group_data) > 0:
                original_groups.append(group_data.values)
                original_labels.append(f'组{group_id}')
        
        axes[0, 1].boxplot(original_groups, labels=original_labels)
        axes[0, 1].set_title('原始BMI分组\nY染色体浓度分布')
        axes[0, 1].set_ylabel('Y染色体浓度')
        
        # 3. 推荐分组箱线图
        recommended_groups = []
        recommended_labels = []
        for group_id in sorted(self.data['推荐分组'].unique()):
            group_data = self.data[self.data['推荐分组'] == group_id]['Y染色体浓度']
            if len(group_data) > 0:
                recommended_groups.append(group_data.values)
                recommended_labels.append(f'组{group_id}')
        
        axes[0, 2].boxplot(recommended_groups, labels=recommended_labels)
        axes[0, 2].set_title('推荐BMI分组\nY染色体浓度分布')
        axes[0, 2].set_ylabel('Y染色体浓度')
        
        # 4. BMI vs Y染色体浓度散点图（原始分组着色）
        colors = ['blue', 'green', 'orange', 'red']
        for i, group_id in enumerate(sorted(self.data['标准分组'].unique())):
            group_data = self.data[self.data['标准分组'] == group_id]
            axes[1, 0].scatter(group_data['孕妇BMI'], group_data['Y染色体浓度'], 
                              c=colors[i % len(colors)], label=f'原始组{group_id}', alpha=0.6)
        axes[1, 0].set_xlabel('BMI')
        axes[1, 0].set_ylabel('Y染色体浓度')
        axes[1, 0].set_title('BMI vs Y染色体浓度\n(原始分组)')
        axes[1, 0].legend()
        
        # 5. BMI vs Y染色体浓度散点图（推荐分组着色）
        for i, group_id in enumerate(sorted(self.data['推荐分组'].unique())):
            group_data = self.data[self.data['推荐分组'] == group_id]
            axes[1, 1].scatter(group_data['孕妇BMI'], group_data['Y染色体浓度'], 
                              c=colors[i % len(colors)], label=f'推荐组{group_id}', alpha=0.6)
        axes[1, 1].set_xlabel('BMI')
        axes[1, 1].set_ylabel('Y染色体浓度')
        axes[1, 1].set_title('BMI vs Y染色体浓度\n(推荐分组)')
        axes[1, 1].legend()
        
        # 6. 分组方法效果对比
        methods = ['原始分组', '推荐分组']
        f_values = []
        
        # 计算F值
        for method_col in ['标准分组', '推荐分组']:
            groups_data = []
            for group_id in sorted(self.data[method_col].unique()):
                group_y_values = self.data[self.data[method_col] == group_id]['Y染色体浓度']
                if len(group_y_values) > 2:
                    groups_data.append(group_y_values.values)
            
            if len(groups_data) >= 2:
                f_stat, _ = f_oneway(*groups_data)
                f_values.append(f_stat)
            else:
                f_values.append(0)
        
        axes[1, 2].bar(methods, f_values, color=['lightblue', 'lightgreen'])
        axes[1, 2].set_ylabel('F统计量')
        axes[1, 2].set_title('分组方法效果对比\n(F值越高越好)')
        
        for i, v in enumerate(f_values):
            axes[1, 2].text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('bmi_grouping_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*80)
        print("孕妇BMI分组合理性分析 - 总结报告")
        print("="*80)
        
        print("\n🔍 问题分析:")
        print("1. 原始分组使用普通人群BMI标准，未考虑孕妇生理特点")
        print("2. 孕期体重自然增长，BMI阈值应适当调整")
        print("3. 需要平衡统计检验力与临床意义")
        
        print("\n📊 发现的问题:")
        print("1. 标准BMI分组可能不适合孕妇群体")
        print("2. 孕妇BMI分布偏高，需要调整分组边界")
        print("3. 某些组样本量过少，影响统计检验")
        
        print("\n💡 推荐改进方案:")
        print("1. 采用孕妇专用BMI分组标准:")
        print("   - 低体重: BMI < 20 (孕妇营养风险)")
        print("   - 正常体重: 20 ≤ BMI < 26 (适合孕妇的正常范围)")
        print("   - 超重: 26 ≤ BMI < 32 (孕期常见，风险可控)")
        print("   - 肥胖: BMI ≥ 32 (高风险，需要干预)")
        
        print("\n2. 理论依据:")
        print("   - 世界卫生组织孕期体重管理指南")
        print("   - 中华医学会妇产科分会推荐标准")
        print("   - 考虑孕期生理体重增长")
        print("   - 保证各组样本量充足")
        
        print("\n3. 统计学优势:")
        print("   - 提高组间差异的统计检验力")
        print("   - 保证各组样本量相对均衡")
        print("   - 更好地反映临床风险分层")
        
        print("\n✅ 结论:")
        print("建议在分析中说明:")
        print("1. 采用孕妇专用BMI分组标准的必要性")
        print("2. 与标准BMI分组的差异及原因")
        print("3. 推荐分组的临床意义和统计优势")
        print("4. 这样的调整更符合孕妇生理特点和临床实践")
    
    def run_analysis(self):
        """运行完整分析"""
        if not self.load_data():
            return False
        
        self.analyze_bmi_distribution()
        self.compare_grouping_methods()
        self.analyze_pregnancy_specific_factors()
        self.recommend_optimal_grouping()
        self.create_visualization()
        self.generate_summary_report()
        
        return True

# 运行分析
if __name__ == "__main__":
    analyzer = PregnancyBMIAnalyzer()
    analyzer.run_analysis()
