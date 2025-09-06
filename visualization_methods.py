import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    plt.show()
    plt.close()

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
    plt.show()
    plt.close()

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
    plt.show()
    plt.close()
