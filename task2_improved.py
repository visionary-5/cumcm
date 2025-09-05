# NIPT问题2改进版：优化BMI分组与最佳检测时点
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedNIPTOptimizer:
    """改进的NIPT BMI分组优化器"""
    
    def __init__(self, data_file='cleaned_data.xlsx'):
        self.data_file = data_file
        self.threshold = 0.04
        
    def load_data(self):
        """加载已清理的数据"""
        try:
            # 读取原始数据重新处理
            original_data = pd.read_excel('附件.xlsx', sheet_name='男胎检测数据')
            
            # 基础数据处理
            # 转换孕周格式
            def convert_gestation_week(week_str):
                if pd.isna(week_str):
                    return np.nan
                week_str = str(week_str).strip().upper()
                
                # 处理 16W+1 格式
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
                
                # 直接是数字的情况
                try:
                    return float(week_str)
                except ValueError:
                    return np.nan
            
            original_data['孕周_数值'] = original_data['检测孕周'].apply(convert_gestation_week)
            
            # 按孕妇分组，计算达标时间
            target_data = []
            
            for woman_code in original_data['孕妇代码'].unique():
                woman_data = original_data[original_data['孕妇代码'] == woman_code].copy()
                woman_data = woman_data.sort_values('孕周_数值')
                
                # 找到第一次达标的时间
                reaching_records = woman_data[woman_data['Y染色体浓度'] >= self.threshold]
                
                if len(reaching_records) > 0:
                    reaching_time = reaching_records['孕周_数值'].iloc[0]
                    target_data.append({
                        '孕妇代码': woman_code,
                        'BMI': woman_data['孕妇BMI'].iloc[0],
                        '年龄': woman_data['年龄'].iloc[0],
                        '达标时间': reaching_time,
                        '最大浓度': woman_data['Y染色体浓度'].max()
                    })
            
            self.analysis_data = pd.DataFrame(target_data)
            print(f"分析数据准备完成：{len(self.analysis_data)}个孕妇")
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def improved_bmi_grouping(self):
        """改进的BMI分组方法"""
        print("\n" + "="*60)
        print("改进的BMI分组分析")
        print("="*60)
        
        # 基于医学标准和数据分布的分组
        bmi_data = self.analysis_data['BMI'].values
        
        # 方法1：基于医学标准的分组
        medical_groups = []
        for bmi in bmi_data:
            if bmi < 25:
                medical_groups.append(1)  # 正常体重
            elif bmi < 28:
                medical_groups.append(2)  # 超重
            elif bmi < 32:
                medical_groups.append(3)  # 轻度肥胖
            elif bmi < 37:
                medical_groups.append(4)  # 中度肥胖
            else:
                medical_groups.append(5)  # 重度肥胖
        
        self.analysis_data['医学分组'] = medical_groups
        
        # 方法2：基于达标时间差异的优化分组
        # 使用分位数分组
        bmi_percentiles = np.percentile(bmi_data, [20, 40, 60, 80])
        
        def get_optimized_group(bmi):
            if bmi <= bmi_percentiles[0]:
                return 1  # 低BMI组
            elif bmi <= bmi_percentiles[1]:
                return 2  # 中低BMI组
            elif bmi <= bmi_percentiles[2]:
                return 3  # 中BMI组
            elif bmi <= bmi_percentiles[3]:
                return 4  # 中高BMI组
            else:
                return 5  # 高BMI组
        
        self.analysis_data['优化分组'] = self.analysis_data['BMI'].apply(get_optimized_group)
        
        # 输出分组结果
        print("医学标准分组结果：")
        medical_summary = self.analysis_data.groupby('医学分组').agg({
            'BMI': ['count', 'min', 'max', 'mean'],
            '达标时间': ['mean', 'std']
        }).round(2)
        print(medical_summary)
        
        print("\n优化分组结果：")
        optimized_summary = self.analysis_data.groupby('优化分组').agg({
            'BMI': ['count', 'min', 'max', 'mean'],
            '达标时间': ['mean', 'std']
        }).round(2)
        print(optimized_summary)
        
        return medical_summary, optimized_summary
    
    def calculate_optimal_timepoints(self):
        """计算最佳检测时点"""
        print("\n" + "="*60)
        print("最佳检测时点计算")
        print("="*60)
        
        recommendations = []
        
        # 对每个医学分组计算最佳时点
        for group_id in sorted(self.analysis_data['医学分组'].unique()):
            group_data = self.analysis_data[self.analysis_data['医学分组'] == group_id]
            
            if len(group_data) == 0:
                continue
                
            # 基础统计
            bmi_min = group_data['BMI'].min()
            bmi_max = group_data['BMI'].max()
            bmi_mean = group_data['BMI'].mean()
            
            reaching_times = group_data['达标时间'].values
            mean_time = np.mean(reaching_times)
            std_time = np.std(reaching_times)
            
            # 计算不同置信水平的推荐时点
            confidence_80 = mean_time + 0.84 * std_time  # 80%置信
            confidence_90 = mean_time + 1.28 * std_time  # 90%置信
            
            # 风险评估
            def calculate_risk(test_time):
                early_risk = np.sum(reaching_times > test_time) / len(reaching_times)  # 检测过早风险
                
                if test_time <= 12:
                    delay_risk = 0.1  # 早期发现，低风险
                elif test_time <= 27:
                    delay_risk = 0.5  # 中期发现，中等风险
                else:
                    delay_risk = 0.9  # 晚期发现，高风险
                
                total_risk = early_risk * 0.7 + delay_risk * 0.3  # 综合风险
                return total_risk, early_risk, delay_risk
            
            # 找到风险最小的时点
            test_times = np.arange(max(10, mean_time - 2*std_time), mean_time + 3*std_time, 0.1)
            test_times = test_times[test_times > 10]  # 至少10周后
            
            if len(test_times) == 0:
                # 如果没有有效的测试时间点，使用默认值
                optimal_time = max(10, mean_time)
                min_risk, early_risk, delay_risk = calculate_risk(optimal_time)
            else:
                risks = [calculate_risk(t)[0] for t in test_times]
                optimal_time = test_times[np.argmin(risks)]
                min_risk, early_risk, delay_risk = calculate_risk(optimal_time)
            
            # 生成推荐
            recommendation = {
                'BMI组': group_id,
                'BMI区间': f"[{bmi_min:.1f}, {bmi_max:.1f}]",
                '样本数': len(group_data),
                '平均BMI': f"{bmi_mean:.1f}",
                '平均达标时间': f"{mean_time:.1f}周",
                '达标时间标准差': f"{std_time:.1f}周",
                '最优检测时点': f"{optimal_time:.1f}周",
                '80%置信时点': f"{confidence_80:.1f}周",
                '90%置信时点': f"{confidence_90:.1f}周",
                '检测过早风险': f"{early_risk:.1%}",
                '延迟发现风险': f"{delay_risk:.1%}",
                '综合风险': f"{min_risk:.1%}"
            }
            
            recommendations.append(recommendation)
            
            print(f"\nBMI组 {group_id} ({bmi_min:.1f}-{bmi_max:.1f}):")
            print(f"  样本数: {len(group_data)}")
            print(f"  平均达标时间: {mean_time:.1f}±{std_time:.1f}周")
            print(f"  推荐检测时点: {optimal_time:.1f}周")
            print(f"  80%置信时点: {confidence_80:.1f}周")
            print(f"  综合风险: {min_risk:.1%}")
        
        self.recommendations = pd.DataFrame(recommendations)
        return self.recommendations
    
    def visualize_improved_analysis(self):
        """改进的可视化分析"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. BMI分布与分组
        ax1 = axes[0, 0]
        ax1.hist(self.analysis_data['BMI'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(25, color='red', linestyle='--', label='正常/超重分界')
        ax1.axvline(28, color='orange', linestyle='--', label='超重/肥胖分界')
        ax1.axvline(32, color='purple', linestyle='--', label='轻度/中度肥胖分界')
        ax1.set_xlabel('BMI')
        ax1.set_ylabel('频数')
        ax1.set_title('BMI分布与医学分组界限')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. BMI vs 达标时间散点图
        ax2 = axes[0, 1]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, group in enumerate(sorted(self.analysis_data['医学分组'].unique())):
            group_data = self.analysis_data[self.analysis_data['医学分组'] == group]
            ax2.scatter(group_data['BMI'], group_data['达标时间'], 
                       c=colors[i], label=f'组{group}', alpha=0.6)
        
        # 添加趋势线
        z = np.polyfit(self.analysis_data['BMI'], self.analysis_data['达标时间'], 1)
        p = np.poly1d(z)
        ax2.plot(self.analysis_data['BMI'], p(self.analysis_data['BMI']), "r--", alpha=0.8)
        
        correlation = self.analysis_data['BMI'].corr(self.analysis_data['达标时间'])
        ax2.set_xlabel('BMI')
        ax2.set_ylabel('达标时间(周)')
        ax2.set_title(f'BMI vs 达标时间 (r={correlation:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 各组达标时间箱线图
        ax3 = axes[0, 2]
        group_data_list = []
        group_labels = []
        for group in sorted(self.analysis_data['医学分组'].unique()):
            group_times = self.analysis_data[self.analysis_data['医学分组'] == group]['达标时间']
            group_data_list.append(group_times)
            bmi_range = self.recommendations[self.recommendations['BMI组'] == group]['BMI区间'].iloc[0]
            group_labels.append(f'组{group}\n{bmi_range}')
        
        box_plot = ax3.boxplot(group_data_list, labels=group_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors[:len(group_data_list)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax3.set_ylabel('达标时间(周)')
        ax3.set_title('各BMI组达标时间分布')
        ax3.grid(True, alpha=0.3)
        
        # 4. 推荐检测时点对比
        ax4 = axes[1, 0]
        groups = self.recommendations['BMI组']
        optimal_times = [float(t.split('周')[0]) for t in self.recommendations['最优检测时点']]
        conf80_times = [float(t.split('周')[0]) for t in self.recommendations['80%置信时点']]
        conf90_times = [float(t.split('周')[0]) for t in self.recommendations['90%置信时点']]
        
        x = np.arange(len(groups))
        width = 0.25
        
        ax4.bar(x - width, optimal_times, width, label='最优时点', color='skyblue')
        ax4.bar(x, conf80_times, width, label='80%置信', color='lightgreen')
        ax4.bar(x + width, conf90_times, width, label='90%置信', color='lightcoral')
        
        ax4.set_xlabel('BMI组')
        ax4.set_ylabel('推荐检测时间(周)')
        ax4.set_title('不同BMI组推荐检测时点')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'组{g}' for g in groups])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 风险分析
        ax5 = axes[1, 1]
        early_risks = [float(r.strip('%'))/100 for r in self.recommendations['检测过早风险']]
        delay_risks = [float(r.strip('%'))/100 for r in self.recommendations['延迟发现风险']]
        total_risks = [float(r.strip('%'))/100 for r in self.recommendations['综合风险']]
        
        ax5.plot(groups, early_risks, 'o-', label='检测过早风险', color='blue')
        ax5.plot(groups, delay_risks, 's-', label='延迟发现风险', color='red')
        ax5.plot(groups, total_risks, '^-', label='综合风险', color='green', linewidth=2)
        
        ax5.set_xlabel('BMI组')
        ax5.set_ylabel('风险概率')
        ax5.set_title('各BMI组风险分析')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 检测时点与风险权衡
        ax6 = axes[1, 2]
        sample_sizes = self.recommendations['样本数']
        risk_colors = total_risks
        
        scatter = ax6.scatter(optimal_times, sample_sizes, c=risk_colors, s=100, 
                             cmap='RdYlGn_r', alpha=0.7)
        
        for i, group in enumerate(groups):
            ax6.annotate(f'组{group}', (optimal_times[i], sample_sizes[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax6.set_xlabel('最优检测时点(周)')
        ax6.set_ylabel('样本数')
        ax6.set_title('检测时点vs样本数vs风险')
        
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('综合风险')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_bmi_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_final_recommendations(self):
        """生成最终推荐方案"""
        print("\n" + "="*80)
        print("最终BMI分组与检测时点推荐方案")
        print("="*80)
        
        # 创建医学标准分组名称映射
        group_names = {
            1: "正常体重组",
            2: "超重组", 
            3: "轻度肥胖组",
            4: "中度肥胖组",
            5: "重度肥胖组"
        }
        
        print("基于医学标准和数据驱动的BMI分组方案：")
        print("-" * 80)
        
        for _, row in self.recommendations.iterrows():
            group_id = row['BMI组']
            group_name = group_names.get(group_id, f"组{group_id}")
            
            print(f"\n{group_name} (BMI {row['BMI区间']}):")
            print(f"  • 样本数量: {row['样本数']}人")
            print(f"  • 平均达标时间: {row['平均达标时间']}")
            print(f"  • 推荐检测时点: {row['最优检测时点']}")
            print(f"  • 保守检测时点: {row['80%置信时点']}")
            print(f"  • 综合风险评估: {row['综合风险']}")
            
            # 临床应用建议
            optimal_week = float(row['最优检测时点'].split('周')[0])
            if optimal_week <= 12:
                risk_level = "低风险 - 早期发现"
            elif optimal_week <= 27:
                risk_level = "中等风险 - 中期发现"
            else:
                risk_level = "高风险 - 晚期发现"
            
            print(f"  • 临床风险等级: {risk_level}")
        
        # 保存推荐方案
        self.recommendations.to_excel('improved_bmi_recommendations.xlsx', index=False)
        print(f"\n推荐方案已保存至: improved_bmi_recommendations.xlsx")
        
        return self.recommendations
    
    def run_improved_analysis(self):
        """运行改进的完整分析"""
        print("NIPT问题2 - 改进版BMI分组与最佳检测时点分析")
        print("="*80)
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 改进的BMI分组
        medical_summary, optimized_summary = self.improved_bmi_grouping()
        
        # 3. 计算最佳检测时点
        recommendations = self.calculate_optimal_timepoints()
        
        # 4. 可视化分析
        self.visualize_improved_analysis()
        
        # 5. 生成最终推荐
        final_recommendations = self.generate_final_recommendations()
        
        print("\n" + "="*80)
        print("改进版分析完成！")
        print("="*80)
        
        return True

# 运行改进的分析
if __name__ == "__main__":
    optimizer = ImprovedNIPTOptimizer()
    success = optimizer.run_improved_analysis()
    
    if success:
        print("\n改进版分析成功完成！")
        print("主要改进：")
        print("1. 基于医学标准的清晰BMI分组")
        print("2. 避免分组区间重叠")
        print("3. 更科学的风险评估模型")
        print("4. 更清晰的可视化展示")
        print("5. 实用的临床应用建议")
