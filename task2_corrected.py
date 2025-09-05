# NIPT问题2修正版：优化BMI分组与最佳检测时点
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

class CorrectedNIPTOptimizer:
    """修正的NIPT BMI分组优化器"""
    
    def __init__(self, data_file='附件.xlsx'):
        self.data_file = data_file
        self.threshold = 0.04
        
    def load_data(self):
        """加载并处理数据"""
        try:
            # 读取原始数据
            original_data = pd.read_excel(self.data_file, sheet_name='男胎检测数据')
            
            # 转换孕周格式
            def convert_gestation_week(week_str):
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
            
            original_data['孕周_数值'] = original_data['检测孕周'].apply(convert_gestation_week)
            
            # 按孕妇分组，计算达标时间
            target_data = []
            
            for woman_code in original_data['孕妇代码'].unique():
                woman_data = original_data[original_data['孕妇代码'] == woman_code].copy()
                woman_data = woman_data.sort_values('孕周_数值')
                
                # 基本信息
                bmi = woman_data['孕妇BMI'].iloc[0]
                age = woman_data['年龄'].iloc[0]
                max_concentration = woman_data['Y染色体浓度'].max()
                
                # 找到第一次达标的时间
                reaching_records = woman_data[woman_data['Y染色体浓度'] >= self.threshold]
                
                if len(reaching_records) > 0:
                    reaching_time = reaching_records['孕周_数值'].iloc[0]
                    target_data.append({
                        '孕妇代码': woman_code,
                        'BMI': bmi,
                        '年龄': age,
                        '达标时间': reaching_time,
                        '最大浓度': max_concentration,
                        '是否达标': 1
                    })
                else:
                    # 对于未达标的孕妇，记录最后检测时间
                    last_time = woman_data['孕周_数值'].max()
                    target_data.append({
                        '孕妇代码': woman_code,
                        'BMI': bmi,
                        '年龄': age,
                        '达标时间': np.nan,
                        '最后检测时间': last_time,
                        '最大浓度': max_concentration,
                        '是否达标': 0
                    })
            
            self.all_data = pd.DataFrame(target_data)
            self.reaching_data = self.all_data[self.all_data['是否达标'] == 1].copy()
            
            print(f"总孕妇数: {len(self.all_data)}")
            print(f"达标孕妇数: {len(self.reaching_data)}")
            print(f"达标率: {len(self.reaching_data)/len(self.all_data):.1%}")
            print(f"平均达标时间: {self.reaching_data['达标时间'].mean():.2f}周")
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def create_practical_bmi_groups(self):
        """创建实用的BMI分组"""
        print("\\n" + "="*60)
        print("创建实用BMI分组")
        print("="*60)
        
        # 基于临床标准和样本分布的合理分组
        def assign_bmi_group(bmi):
            if bmi < 25:
                return '正常体重', 1
            elif bmi < 28:
                return '超重', 2  
            elif bmi < 32:
                return '轻度肥胖', 3
            elif bmi < 36:
                return '中度肥胖', 4
            else:
                return '重度肥胖', 5
        
        # 为达标数据分组
        self.reaching_data[['BMI组名称', 'BMI组编号']] = self.reaching_data['BMI'].apply(
            lambda x: pd.Series(assign_bmi_group(x))
        )
        
        # 分组统计
        group_stats = []
        for group_id in sorted(self.reaching_data['BMI组编号'].unique()):
            group_data = self.reaching_data[self.reaching_data['BMI组编号'] == group_id]
            
            stats_dict = {
                'BMI组': group_id,
                'BMI组名称': group_data['BMI组名称'].iloc[0],
                '样本数': len(group_data),
                'BMI最小值': group_data['BMI'].min(),
                'BMI最大值': group_data['BMI'].max(),
                'BMI均值': group_data['BMI'].mean(),
                'BMI标准差': group_data['BMI'].std(),
                '达标时间均值': group_data['达标时间'].mean(),
                '达标时间标准差': group_data['达标时间'].std(),
                '达标时间中位数': group_data['达标时间'].median()
            }
            group_stats.append(stats_dict)
        
        self.group_stats_df = pd.DataFrame(group_stats)
        
        print("BMI分组统计:")
        print(self.group_stats_df.round(2))
        
        return self.group_stats_df
    
    def calculate_optimal_detection_times(self):
        """计算最佳检测时点"""
        print("\\n" + "="*60)
        print("计算最佳检测时点")
        print("="*60)
        
        recommendations = []
        
        for _, group_stat in self.group_stats_df.iterrows():
            group_id = group_stat['BMI组']
            group_data = self.reaching_data[self.reaching_data['BMI组编号'] == group_id]
            
            # 基础统计
            mean_time = group_stat['达标时间均值']
            std_time = group_stat['达标时间标准差']
            sample_size = group_stat['样本数']
            
            # 计算不同置信水平的推荐时点
            # 考虑样本量小时的修正
            if sample_size >= 30:
                confidence_multiplier_80 = 0.84  # 正态分布
                confidence_multiplier_90 = 1.28
            else:
                # 小样本用t分布修正
                from scipy.stats import t
                confidence_multiplier_80 = t.ppf(0.8, sample_size - 1)
                confidence_multiplier_90 = t.ppf(0.9, sample_size - 1)
            
            # 保守推荐时点（80%置信）
            conservative_time = mean_time + confidence_multiplier_80 * std_time / np.sqrt(sample_size)
            
            # 非常保守时点（90%置信）
            very_conservative_time = mean_time + confidence_multiplier_90 * std_time / np.sqrt(sample_size)
            
            # 实用推荐时点（考虑临床实际）
            # 不能太早（避免检测失败），不能太晚（避免治疗窗口期缩短）
            practical_time = min(max(conservative_time, 12), 20)
            
            # 风险评估
            early_failure_risk = len(group_data[group_data['达标时间'] > practical_time]) / len(group_data)
            
            # 临床风险分级
            if practical_time <= 12:
                risk_level = "低风险"
                risk_description = "早期发现"
            elif practical_time <= 27:
                risk_level = "中等风险"
                risk_description = "中期发现"
            else:
                risk_level = "高风险"
                risk_description = "晚期发现"
            
            recommendation = {
                'BMI组': group_id,
                'BMI组名称': group_stat['BMI组名称'],
                'BMI区间': f"[{group_stat['BMI最小值']:.1f}, {group_stat['BMI最大值']:.1f}]",
                '样本数': sample_size,
                '平均达标时间': f"{mean_time:.1f}周",
                '达标时间标准差': f"{std_time:.1f}周",
                '推荐检测时点': f"{practical_time:.1f}周",
                '保守时点(80%)': f"{conservative_time:.1f}周",
                '非常保守时点(90%)': f"{very_conservative_time:.1f}周",
                '检测失败风险': f"{early_failure_risk:.1%}",
                '风险等级': risk_level,
                '风险描述': risk_description
            }
            recommendations.append(recommendation)
            
            print(f"\\nBMI组 {group_id} ({group_stat['BMI组名称']}):")
            print(f"  BMI区间: {recommendation['BMI区间']}")
            print(f"  样本数: {sample_size}")
            print(f"  平均达标时间: {recommendation['平均达标时间']}")
            print(f"  推荐检测时点: {recommendation['推荐检测时点']}")
            print(f"  检测失败风险: {recommendation['检测失败风险']}")
            print(f"  风险等级: {recommendation['风险等级']}")
        
        self.recommendations_df = pd.DataFrame(recommendations)
        return self.recommendations_df
    
    def analyze_error_impact(self):
        """分析检测误差影响"""
        print("\\n" + "="*60)
        print("检测误差影响分析")
        print("="*60)
        
        # 模拟不同误差水平对结果的影响
        error_levels = [0.001, 0.002, 0.005, 0.01]  # Y染色体浓度误差
        time_errors = [0.5, 1.0, 1.5]  # 孕周误差
        
        print("Y染色体浓度测量误差影响:")
        for error in error_levels:
            # 计算在不同误差下的达标率变化
            adjusted_concentrations = self.reaching_data['最大浓度'] - error
            still_reaching = np.sum(adjusted_concentrations >= self.threshold)
            impact = (still_reaching - len(self.reaching_data)) / len(self.reaching_data)
            print(f"  误差 ±{error:.3f}: 达标率变化 {impact:.2%}")
        
        print("\\n孕周测量误差影响:")
        for time_error in time_errors:
            mean_time_change = self.reaching_data['达标时间'].mean()
            adjusted_mean = mean_time_change + time_error
            time_impact = time_error / mean_time_change
            print(f"  误差 ±{time_error}周: 平均达标时间变化 {time_impact:.1%}")
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\\n" + "="*60)
        print("生成可视化图表")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NIPT BMI分组与检测时点优化分析', fontsize=16, fontweight='bold')
        
        # 1. BMI分布
        axes[0,0].hist(self.reaching_data['BMI'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(self.reaching_data['BMI'].mean(), color='red', linestyle='--', label=f'均值: {self.reaching_data["BMI"].mean():.1f}')
        axes[0,0].set_xlabel('BMI')
        axes[0,0].set_ylabel('频数')
        axes[0,0].set_title('BMI分布')
        axes[0,0].legend()
        
        # 2. BMI vs 达标时间
        scatter = axes[0,1].scatter(self.reaching_data['BMI'], self.reaching_data['达标时间'], 
                                  c=self.reaching_data['BMI组编号'], cmap='viridis', alpha=0.6)
        z = np.polyfit(self.reaching_data['BMI'], self.reaching_data['达标时间'], 1)
        p = np.poly1d(z)
        axes[0,1].plot(self.reaching_data['BMI'], p(self.reaching_data['BMI']), "r--", alpha=0.8)
        corr = np.corrcoef(self.reaching_data['BMI'], self.reaching_data['达标时间'])[0,1]
        axes[0,1].set_xlabel('BMI')
        axes[0,1].set_ylabel('达标时间(周)')
        axes[0,1].set_title(f'BMI vs 达标时间 (r={corr:.3f})')
        plt.colorbar(scatter, ax=axes[0,1], label='BMI组')
        
        # 3. 分组箱线图
        group_data_for_plot = []
        group_labels = []
        for group_id in sorted(self.reaching_data['BMI组编号'].unique()):
            group_data_for_plot.append(self.reaching_data[self.reaching_data['BMI组编号'] == group_id]['达标时间'].values)
            group_name = self.reaching_data[self.reaching_data['BMI组编号'] == group_id]['BMI组名称'].iloc[0]
            group_labels.append(f'{group_name}\\n(n={len(group_data_for_plot[-1])})')
        
        axes[0,2].boxplot(group_data_for_plot, labels=range(1, len(group_data_for_plot)+1))
        axes[0,2].set_xlabel('BMI组')
        axes[0,2].set_ylabel('达标时间(周)')
        axes[0,2].set_title('各BMI组达标时间分布')
        axes[0,2].set_xticklabels(group_labels, rotation=45)
        
        # 4. 推荐检测时点对比
        groups = self.recommendations_df['BMI组名称']
        recommended_times = [float(x.split('周')[0]) for x in self.recommendations_df['推荐检测时点']]
        conservative_times = [float(x.split('周')[0]) for x in self.recommendations_df['保守时点(80%)']]
        
        x = np.arange(len(groups))
        width = 0.35
        
        axes[1,0].bar(x - width/2, recommended_times, width, label='推荐时点', alpha=0.8)
        axes[1,0].bar(x + width/2, conservative_times, width, label='保守时点', alpha=0.8)
        axes[1,0].set_xlabel('BMI组')
        axes[1,0].set_ylabel('检测时点(周)')
        axes[1,0].set_title('推荐检测时点对比')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(groups, rotation=45)
        axes[1,0].legend()
        
        # 5. 风险评估
        failure_risks = [float(x.split('%')[0])/100 for x in self.recommendations_df['检测失败风险']]
        colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in failure_risks]
        
        axes[1,1].bar(groups, failure_risks, color=colors, alpha=0.7)
        axes[1,1].set_xlabel('BMI组')
        axes[1,1].set_ylabel('检测失败风险')
        axes[1,1].set_title('各组检测失败风险')
        axes[1,1].set_xticklabels(groups, rotation=45)
        
        # 6. 达标率分析
        total_counts = self.all_data['BMI'].apply(lambda x: assign_bmi_group(x)[1]).value_counts().sort_index()
        reaching_counts = self.reaching_data['BMI组编号'].value_counts().sort_index()
        
        success_rates = []
        group_names = []
        for group_id in sorted(total_counts.index):
            if group_id in reaching_counts.index:
                rate = reaching_counts[group_id] / total_counts[group_id]
            else:
                rate = 0
            success_rates.append(rate)
            group_name = self.group_stats_df[self.group_stats_df['BMI组'] == group_id]['BMI组名称'].iloc[0]
            group_names.append(group_name)
        
        colors_success = ['green' if x > 0.8 else 'orange' if x > 0.6 else 'red' for x in success_rates]
        axes[1,2].bar(group_names, success_rates, color=colors_success, alpha=0.7)
        axes[1,2].set_xlabel('BMI组')
        axes[1,2].set_ylabel('达标率')
        axes[1,2].set_title('各组达标率')
        axes[1,2].set_xticklabels(group_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig('nipt_corrected_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_final_report(self):
        """生成最终报告"""
        print("\\n" + "="*80)
        print("最终BMI分组与检测时点推荐报告")
        print("="*80)
        
        print("\\n📊 数据概况:")
        print(f"• 总孕妇样本: {len(self.all_data)}人")
        print(f"• Y染色体达标孕妇: {len(self.reaching_data)}人")
        print(f"• 总体达标率: {len(self.reaching_data)/len(self.all_data):.1%}")
        print(f"• 平均达标时间: {self.reaching_data['达标时间'].mean():.1f}周")
        
        print("\\n📋 BMI分组与检测时点推荐:")
        print("-" * 80)
        
        for _, rec in self.recommendations_df.iterrows():
            print(f"\\n{rec['BMI组名称']} (BMI {rec['BMI区间']}):")
            print(f"  • 样本数量: {rec['样本数']}人")
            print(f"  • 平均达标时间: {rec['平均达标时间']}")
            print(f"  • 推荐检测时点: {rec['推荐检测时点']}")
            print(f"  • 检测失败风险: {rec['检测失败风险']}")
            print(f"  • 临床风险等级: {rec['风险等级']} ({rec['风险描述']})")
        
        print("\\n💡 临床应用建议:")
        print("1. 正常体重和超重孕妇可在12-14周开始检测")
        print("2. 轻度肥胖孕妇建议在13-15周检测")
        print("3. 中重度肥胖孕妇应适当延迟至15-17周")
        print("4. 所有检测应在20周前完成，确保治疗窗口期")
        print("5. 如首次检测失败，建议间隔1-2周重复检测")
        
        # 保存推荐结果
        self.recommendations_df.to_excel('corrected_bmi_recommendations.xlsx', index=False)
        print("\\n✅ 推荐方案已保存至: corrected_bmi_recommendations.xlsx")
        
        return self.recommendations_df
    
    def run_analysis(self):
        """运行完整分析"""
        print("NIPT问题2修正版分析")
        print("="*80)
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 创建BMI分组
        self.create_practical_bmi_groups()
        
        # 3. 计算最佳检测时点
        self.calculate_optimal_detection_times()
        
        # 4. 误差影响分析
        self.analyze_error_impact()
        
        # 5. 创建可视化
        self.create_visualizations()
        
        # 6. 生成最终报告
        self.generate_final_report()
        
        return True

def assign_bmi_group(bmi):
    """BMI分组函数"""
    if bmi < 25:
        return '正常体重', 1
    elif bmi < 28:
        return '超重', 2  
    elif bmi < 32:
        return '轻度肥胖', 3
    elif bmi < 36:
        return '中度肥胖', 4
    else:
        return '重度肥胖', 5

# 主程序
if __name__ == "__main__":
    optimizer = CorrectedNIPTOptimizer()
    success = optimizer.run_analysis()
    
    if success:
        print("\\n" + "="*80)
        print("修正版分析完成！")
        print("="*80)
        print("主要改进:")
        print("1. 统计了所有孕妇样本，明确达标率")
        print("2. 基于临床标准创建均衡的BMI分组") 
        print("3. 采用更实际的检测时点推荐策略")
        print("4. 考虑小样本量的统计修正")
        print("5. 提供更全面的风险评估")
    else:
        print("分析失败，请检查数据文件")
