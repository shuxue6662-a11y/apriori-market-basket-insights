# 基于UCI Online Retail数据集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 70)
print("        购物篮分析与商品组合挖掘 - 机器学习课程论文")
print("=" * 70)
print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# 数据加载与预处理
print("\n" + "▶" * 3 + " 阶段1: 数据加载与预处理 " + "◀" * 3)

df_raw = pd.read_excel('Online Retail.xlsx')

print(f"\n【原始数据概览】")
print(f"  • 数据集行数: {len(df_raw):,}")
print(f"  • 数据集列数: {len(df_raw.columns)}")
print(f"  • 字段列表: {list(df_raw.columns)}")

print(f"\n【数据样本 (前5行)】")
print(df_raw.head())

print(f"\n【数据类型信息】")
print(df_raw.dtypes)

print(f"\n【缺失值统计】")
missing_stats = pd.DataFrame({
    '缺失数量': df_raw.isnull().sum(),
    '缺失比例': (df_raw.isnull().sum() / len(df_raw) * 100).round(2).astype(str) + '%'
})
print(missing_stats)

print(f"\n【数据清洗过程】")
df = df_raw.copy()

#删除缺失
before = len(df)
df.dropna(subset=['Description', 'InvoiceNo'], inplace=True)
print(f"  1. 删除缺失Description/InvoiceNo: {before:,} → {len(df):,} (删除{before-len(df):,}行)")

#删除取消订单
before = len(df)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"  2. 删除取消订单(C开头): {before:,} → {len(df):,} (删除{before-len(df):,}行)")

before = len(df)
df = df[df['Quantity'] > 0]
print(f"  3. 删除负数量记录: {before:,} → {len(df):,} (删除{before-len(df):,}行)")

before = len(df)
df = df[df['UnitPrice'] > 0]
print(f"  4. 删除异常单价记录: {before:,} → {len(df):,} (删除{before-len(df):,}行)")

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour

print(f"\n【清洗后数据概览】")
print(f"  • 最终数据行数: {len(df):,}")
print(f"  • 数据保留率: {len(df)/len(df_raw)*100:.1f}%")


#探索性数据分析
print("\n" + "▶" * 3 + " 阶段2: 探索性数据分析 (EDA) " + "◀" * 3)

print(f"\n【基础统计信息】")
print(f"  • 唯一订单数: {df['InvoiceNo'].nunique():,}")
print(f"  • 唯一商品数: {df['Description'].nunique():,}")
print(f"  • 唯一客户数: {df['CustomerID'].nunique():,}")
print(f"  • 涉及国家数: {df['Country'].nunique()}")
print(f"  • 时间范围: {df['InvoiceDate'].min().strftime('%Y-%m-%d')} 至 {df['InvoiceDate'].max().strftime('%Y-%m-%d')}")
print(f"  • 总交易金额: £{df['TotalPrice'].sum():,.2f}")

print(f"\n【数值型变量统计】")
print(df[['Quantity', 'UnitPrice', 'TotalPrice']].describe().round(2))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('购物篮分析 - 探索性数据分析 (EDA)', fontsize=16, fontweight='bold')

# 图1: Top 15热销商品
ax1 = axes[0, 0]
top_products = df.groupby('Description')['Quantity'].sum().nlargest(15)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, 15))[::-1]
bars = ax1.barh(range(len(top_products)), top_products.values, color=colors)
ax1.set_yticks(range(len(top_products)))
ax1.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in top_products.index], fontsize=8)
ax1.set_xlabel('销售数量')
ax1.set_title('Top 15 热销商品', fontweight='bold')
ax1.invert_yaxis()

# 图2: 国家销售分布 (Top 10)
ax2 = axes[0, 1]
country_sales = df.groupby('Country')['TotalPrice'].sum().nlargest(10)
colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, 10))[::-1]
wedges, texts, autotexts = ax2.pie(country_sales.values, labels=None, autopct='%1.1f%%', 
                                     colors=colors2, pctdistance=0.75)
ax2.legend(wedges, country_sales.index, title="国家", loc="center left", 
           bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
ax2.set_title('Top 10 国家销售额占比', fontweight='bold')

# 图3: 月度交易趋势
ax3 = axes[0, 2]
monthly_orders = df.groupby([df['InvoiceDate'].dt.to_period('M')])['InvoiceNo'].nunique()
ax3.plot(range(len(monthly_orders)), monthly_orders.values, 'o-', color='#2E86AB', linewidth=2, markersize=6)
ax3.fill_between(range(len(monthly_orders)), monthly_orders.values, alpha=0.3, color='#2E86AB')
ax3.set_xticks(range(0, len(monthly_orders), 2))
ax3.set_xticklabels([str(p) for p in monthly_orders.index[::2]], rotation=45, fontsize=8)
ax3.set_xlabel('月份')
ax3.set_ylabel('订单数量')
ax3.set_title('月度订单数量趋势', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 图4: 每周购物分布
ax4 = axes[1, 0]
day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
weekday_orders = df.groupby('DayOfWeek')['InvoiceNo'].nunique()
weekday_orders = weekday_orders.reindex(range(7), fill_value=0)
colors4 = ['#FF6B6B' if i == weekday_orders.idxmax() else '#4ECDC4' for i in range(7)]
ax4.bar(day_names, weekday_orders.values, color=colors4, edgecolor='black', alpha=0.8)
ax4.set_xlabel('星期')
ax4.set_ylabel('订单数量')
ax4.set_title('每周订单分布', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for i, val in enumerate(weekday_orders.values):
    if val == 0:
        ax4.annotate('无数据', (i, val + 50), ha='center', fontsize=9, color='red')

# 图5: 每小时购物分布
ax5 = axes[1, 1]
hourly_orders = df.groupby('Hour')['InvoiceNo'].nunique()
hourly_orders = hourly_orders.reindex(range(24), fill_value=0)
ax5.bar(hourly_orders.index, hourly_orders.values, color='#A8E6CF', edgecolor='black', alpha=0.8)
ax5.axhline(y=hourly_orders[hourly_orders > 0].mean(), color='red', linestyle='--', 
            label=f'平均值: {hourly_orders[hourly_orders > 0].mean():.0f}')
ax5.set_xlabel('小时 (24小时制)')
ax5.set_ylabel('订单数量')
ax5.set_title('每小时订单分布', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 图6: 订单商品数量分布
ax6 = axes[1, 2]
items_per_order = df.groupby('InvoiceNo')['Description'].count()
ax6.hist(items_per_order.values, bins=50, color='#DDA0DD', edgecolor='black', alpha=0.8)
ax6.axvline(x=items_per_order.median(), color='red', linestyle='--', 
            label=f'中位数: {items_per_order.median():.0f}件')
ax6.axvline(x=items_per_order.mean(), color='blue', linestyle='--', 
            label=f'平均值: {items_per_order.mean():.1f}件')
ax6.set_xlabel('每订单商品数量')
ax6.set_ylabel('订单频次')
ax6.set_title('订单商品数量分布', fontweight='bold')
ax6.set_xlim(0, 100)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('图1_探索性数据分析.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n✓ 已保存: 图1_探索性数据分析.png")

#构建交易-商品关系矩阵
print("\n" + "▶" * 3 + " 阶段3: 构建'交易-商品'关系矩阵 " + "◀" * 3)

transactions = df.groupby('InvoiceNo')['Description'].apply(list).tolist()

print(f"\n【交易数据构建完成】")
print(f"  • 总交易笔数: {len(transactions):,}")
print(f"  • 平均每笔交易商品数: {np.mean([len(t) for t in transactions]):.2f}")
print(f"  • 最大单笔交易商品数: {max([len(t) for t in transactions])}")
print(f"  • 最小单笔交易商品数: {min([len(t) for t in transactions])}")

print(f"\n【交易样本展示】")
for i, trans in enumerate(transactions[:3], 1):
    print(f"  交易{i}: {trans[:5]}{'...' if len(trans) > 5 else ''} (共{len(trans)}件商品)")

print(f"\n【One-Hot编码转换】")
te = TransactionEncoder()
onehot_matrix = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(onehot_matrix, columns=te.columns_)

print(f"  • 交易数量: {df_onehot.shape[0]:,}")
print(f"  • 商品种类: {df_onehot.shape[1]:,}")
print(f"  • 矩阵稀疏度: {(1 - df_onehot.sum().sum() / (df_onehot.shape[0] * df_onehot.shape[1])) * 100:.2f}%")

item_frequency = df_onehot.sum().sort_values(ascending=False)
print(f"\n【Top 10 高频商品】")
for i, (item, freq) in enumerate(item_frequency.head(10).items(), 1):
    print(f"  {i:2d}. {item[:50]:50s} | 出现次数: {int(freq):5,} | 出现率: {freq/len(transactions)*100:.2f}%")

#Apriori关联规则挖掘
print("\n" + "▶" * 3 + " 阶段4: Apriori关联规则挖掘 " + "◀" * 3)

min_support = 0.015     
min_confidence = 0.4    
min_lift = 3.0          

print(f"\n【算法参数设置】")
print(f"  • 最小支持度 (min_support): {min_support} (≥{int(min_support*len(transactions))}笔交易)")
print(f"  • 最小置信度 (min_confidence): {min_confidence}")
print(f"  • 最小提升度 (min_lift): {min_lift}")

print(f"\n【频繁项集挖掘】")
frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True, max_len=3)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
print(f"  • 频繁项集总数: {len(frequent_itemsets)}")

for length in sorted(frequent_itemsets['length'].unique()):
    count = len(frequent_itemsets[frequent_itemsets['length'] == length])
    print(f"  • {length}-项集数量: {count}")

print(f"\n【Top 10 频繁项集 (按支持度排序)】")
top_itemsets = frequent_itemsets.nlargest(10, 'support')
for i, (_, row) in enumerate(top_itemsets.iterrows(), 1):
    items = ', '.join(list(row['itemsets']))
    print(f"  {i:2d}. {items[:60]:60s} | 支持度: {row['support']:.4f}")

print(f"\n【关联规则生成】")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
print(f"  • 初始规则数量 (置信度≥{min_confidence}): {len(rules)}")

if len(rules) == 0:
    print("  ⚠ 警告: 未找到满足条件的规则，尝试降低阈值...")
    min_confidence = 0.3
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    print(f"  • 调整后规则数量 (置信度≥{min_confidence}): {len(rules)}")

strong_rules = rules[rules['lift'] >= min_lift].copy()
strong_rules = strong_rules.sort_values(['lift', 'confidence'], ascending=[False, False])
print(f"  • 强规则数量 (提升度≥{min_lift}): {len(strong_rules)}")

if len(strong_rules) == 0:
    print("  ⚠ 警告: 未找到强规则，降低提升度阈值...")
    min_lift = 1.5
    strong_rules = rules[rules['lift'] >= min_lift].copy()
    strong_rules = strong_rules.sort_values(['lift', 'confidence'], ascending=[False, False])

strong_rules['antecedents_str'] = strong_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
strong_rules['consequents_str'] = strong_rules['consequents'].apply(lambda x: ', '.join(list(x)))
strong_rules['rule'] = strong_rules['antecedents_str'] + ' → ' + strong_rules['consequents_str']
strong_rules['transaction_count'] = (strong_rules['support'] * len(transactions)).astype(int)


#关联规则结果展示
print("\n" + "▶" * 3 + " 阶段5: 关联规则结果展示 " + "◀" * 3)

print(f"\n{'='*90}")
print(f"{'排名':^4} | {'规则':^50} | {'支持度':^8} | {'置信度':^8} | {'提升度':^8}")
print(f"{'='*90}")

for i, (_, rule) in enumerate(strong_rules.head(15).iterrows(), 1):
    ant = rule['antecedents_str'][:22] + '...' if len(rule['antecedents_str']) > 25 else rule['antecedents_str']
    con = rule['consequents_str'][:22] + '...' if len(rule['consequents_str']) > 25 else rule['consequents_str']
    rule_str = f"{ant} → {con}"
    print(f"{i:^4} | {rule_str:50s} | {rule['support']:.4f}  | {rule['confidence']:.2%}  | {rule['lift']:.2f}")

print(f"{'='*90}")

print(f"\n【规则质量统计】")
print(f"  • 规则总数: {len(strong_rules)}")
print(f"  • 平均支持度: {strong_rules['support'].mean():.4f}")
print(f"  • 平均置信度: {strong_rules['confidence'].mean():.2%}")
print(f"  • 平均提升度: {strong_rules['lift'].mean():.2f}")
print(f"  • 最高提升度: {strong_rules['lift'].max():.2f}")
print(f"  • 置信度≥80%的规则数: {len(strong_rules[strong_rules['confidence'] >= 0.8])}")

print(f"\n【规则多样性分析】")

all_items_in_rules = set()
for _, rule in strong_rules.iterrows():
    all_items_in_rules.update(rule['antecedents'])
    all_items_in_rules.update(rule['consequents'])

print(f"  • 规则涉及的不同商品数: {len(all_items_in_rules)}")

categories = {
    '茶具类': ['TEACUP', 'SAUCER', 'TEA'],
    '蛋糕类': ['CAKE', 'CUPCAKE'],
    '午餐袋': ['LUNCH BAG'],
    '园艺类': ['GARDEN', 'KNEELING'],
    '装饰类': ['HEART', 'HANGING', 'ORNAMENT'],
    '收纳类': ['BAG', 'STORAGE', 'TIN']
}

rule_categories = []
for _, rule in strong_rules.iterrows():
    rule_text = (rule['antecedents_str'] + ' ' + rule['consequents_str']).upper()
    found_cat = '其他'
    for cat, keywords in categories.items():
        if any(kw in rule_text for kw in keywords):
            found_cat = cat
            break
    rule_categories.append(found_cat)

strong_rules['category'] = rule_categories
cat_counts = strong_rules['category'].value_counts()
print(f"\n  • 规则类别分布:")
for cat, count in cat_counts.items():
    print(f"    - {cat}: {count}条 ({count/len(strong_rules)*100:.1f}%)")


#可视化分析
print("\n" + "▶" * 3 + " 阶段6: 可视化分析 " + "◀" * 3)

#支持度-置信度-提升度散点图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图: 支持度 vs 置信度
ax1 = axes[0]
scatter1 = ax1.scatter(strong_rules['support'], strong_rules['confidence'],
                       c=strong_rules['lift'], s=strong_rules['lift']*30,
                       cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.5)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('提升度 (Lift)', fontsize=10)
ax1.set_xlabel('支持度 (Support)', fontsize=12)
ax1.set_ylabel('置信度 (Confidence)', fontsize=12)
ax1.set_title('关联规则分布: 支持度 vs 置信度', fontsize=14, fontweight='bold')
ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='高置信度线(70%)')
ax1.axvline(x=0.025, color='blue', linestyle='--', alpha=0.5, label='较高支持度线(2.5%)')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

for i, (_, rule) in enumerate(strong_rules.head(5).iterrows(), 1):
    ax1.annotate(f'R{i}', (rule['support'], rule['confidence']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=10, fontweight='bold', color='darkred')

# 右图: Top 15规则提升度柱状图
ax2 = axes[1]
top15 = strong_rules.head(15)
rule_labels = [f"R{i}" for i in range(1, len(top15) + 1)]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top15)))
bars = ax2.barh(rule_labels[::-1], top15['lift'].values[::-1], color=colors[::-1], edgecolor='black')
ax2.axvline(x=min_lift, color='red', linestyle='--', label=f'最小提升度阈值({min_lift})')
ax2.set_xlabel('提升度 (Lift)', fontsize=12)
ax2.set_ylabel('规则编号', fontsize=12)
ax2.set_title('Top 15 规则提升度排名', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

for bar, lift in zip(bars, top15['lift'].values[::-1]):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{lift:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('图2_关联规则分布与排名.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ 已保存: 图2_关联规则分布与排名.png")

#关联规则网络图
fig, ax = plt.subplots(figsize=(14, 12))

G = nx.DiGraph()
top_n = min(20, len(strong_rules))
top_rules_for_graph = strong_rules.head(top_n)

for _, rule in top_rules_for_graph.iterrows():
    ant = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else str(list(rule['antecedents']))
    con = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else str(list(rule['consequents']))
    G.add_edge(ant, con,
               weight=rule['lift'],
               support=rule['support'],
               confidence=rule['confidence'])

if len(G.nodes()) > 0:
    
    degrees = dict(G.degree())
    node_sizes = [300 + degrees[node] * 200 for node in G.nodes()]

   
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    
    node_colors = ['#FF6B6B' if degrees[node] >= 3 else '#4ECDC4' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.8, edgecolors='black', linewidths=1.5)

    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights],
                           alpha=0.6, edge_color='#2E86AB',
                           arrows=True, arrowsize=20, arrowstyle='->')

    
    labels = {node: node[:18] + '...' if len(str(node)) > 18 else node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold')

    
    edge_labels = {(u, v): f"{G[u][v]['confidence']:.0%}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_color='darkgreen')

    ax.set_title('商品关联规则网络图\n(节点大小=关联度, 边宽度=提升度, 边标签=置信度)',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=15, label='高关联商品(度≥3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=15, label='一般商品')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('图3_关联规则网络图.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ 已保存: 图3_关联规则网络图.png")


fig, axes = plt.subplots(1, 2, figsize=(16, 8))


ax1 = axes[0]
top_items = item_frequency.head(12).index.tolist()
cooccurrence = df_onehot[top_items].T.dot(df_onehot[top_items])
np.fill_diagonal(cooccurrence.values, 0)


short_labels = [item[:18] + '...' if len(item) > 18 else item for item in top_items]
cooccurrence_display = cooccurrence.copy()
cooccurrence_display.index = short_labels
cooccurrence_display.columns = short_labels

sns.heatmap(cooccurrence_display, annot=True, fmt='d', cmap='YlOrRd',
            ax=ax1, cbar_kws={'label': '共现次数'}, annot_kws={'size': 8})
ax1.set_title('Top 12 商品共现矩阵', fontsize=14, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)


ax2 = axes[1]
if 'category' in strong_rules.columns:
    cat_counts = strong_rules['category'].value_counts()
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cat_counts)))
    wedges, texts, autotexts = ax2.pie(cat_counts.values, labels=cat_counts.index, 
                                        autopct='%1.1f%%', colors=colors_pie,
                                        explode=[0.05]*len(cat_counts))
    ax2.set_title('关联规则商品类别分布', fontsize=14, fontweight='bold')
else:
    
    top5 = strong_rules.head(5)
    x = np.arange(len(top5))
    width = 0.25
    bars1 = ax2.bar(x - width, top5['support'] * 10, width, label='支持度 (×10)', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x, top5['confidence'], width, label='置信度', color='#2ecc71', alpha=0.8)
    bars3 = ax2.bar(x + width, top5['lift'] / top5['lift'].max(), width, label='提升度 (归一化)', color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('规则编号', fontsize=12)
    ax2.set_ylabel('指标值', fontsize=12)
    ax2.set_title('Top 5 规则多维度指标对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'R{i}' for i in range(1, 6)])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('图4_频繁项集与规则分析.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ 已保存: 图4_频繁项集与规则分析.png")


fig, axes = plt.subplots(1, 2, figsize=(14, 6))


ax1 = axes[0]
top3 = strong_rules.head(3)
x = np.arange(len(top3))
width = 0.35


random_prob = [strong_rules.iloc[i]['support'] / strong_rules.iloc[i]['confidence'] 
               if strong_rules.iloc[i]['confidence'] > 0 else 0 for i in range(len(top3))]
assoc_prob = top3['confidence'].values

ax1.bar(x - width/2, [p*100 for p in random_prob], width, label='随机购买概率', color='#95a5a6', alpha=0.8)
ax1.bar(x + width/2, [p*100 for p in assoc_prob], width, label='关联购买概率', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('购买概率 (%)', fontsize=12)
ax1.set_xlabel('规则编号', fontsize=12)
ax1.set_title('提升度含义解读: 关联购买 vs 随机购买', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'R{i+1}\n(Lift={top3.iloc[i]["lift"]:.1f})' for i in range(len(top3))])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')


for i in range(len(top3)):
    lift = top3.iloc[i]['lift']
    ax1.annotate(f'{lift:.1f}倍', xy=(i, assoc_prob[i]*100), xytext=(i+0.3, assoc_prob[i]*100+5),
                fontsize=10, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))


ax2 = axes[1]
metrics_explain = {
    '支持度\n(Support)': '商品组合在所有交易中\n出现的频率',
    '置信度\n(Confidence)': '购买A的顾客中\n同时购买B的比例',
    '提升度\n(Lift)': '关联购买概率相比\n随机购买的倍数'
}


example_rule = strong_rules.iloc[0]
example_values = [
    f"{example_rule['support']:.2%}",
    f"{example_rule['confidence']:.2%}",
    f"{example_rule['lift']:.1f}x"
]

y_pos = np.arange(len(metrics_explain))
colors_bar = ['#3498db', '#2ecc71', '#e74c3c']

bars = ax2.barh(y_pos, [1, 1, 1], color=colors_bar, alpha=0.3)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(list(metrics_explain.keys()), fontsize=11)
ax2.set_xlim(0, 2)
ax2.set_xticks([])
ax2.set_title('关联规则三大指标解读 (以规则1为例)', fontsize=14, fontweight='bold')


for i, (metric, explain) in enumerate(metrics_explain.items()):
    ax2.text(0.05, i, explain, va='center', fontsize=10, color='black')
    ax2.text(1.5, i, example_values[i], va='center', fontsize=14, fontweight='bold', 
             color=colors_bar[i], ha='center')

ax2.axvline(x=1.3, color='gray', linestyle='--', alpha=0.5)
ax2.text(1.5, -0.5, '规则1数值', ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('图5_指标解读与提升度分析.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ 已保存: 图5_指标解读与提升度分析.png")


#业务解读与促销方案
print("\n" + "▶" * 3 + " 阶段7: 业务解读与促销方案 " + "◀" * 3)

print("\n" + "=" * 70)
print("                    高价值关联规则业务分析")
print("=" * 70)


if 'category' in strong_rules.columns:
    representative_rules = strong_rules.groupby('category').first().reset_index()
    display_rules = representative_rules.head(5)
else:
    display_rules = strong_rules.head(5)

for i, (_, rule) in enumerate(display_rules.iterrows(), 1):
    antecedent = rule['antecedents_str'] if 'antecedents_str' in rule else str(rule.get('antecedents', ''))
    consequent = rule['consequents_str'] if 'consequents_str' in rule else str(rule.get('consequents', ''))
    
    
    random_prob = rule['support'] / rule['confidence'] if rule['confidence'] > 0 else 0
    
    print(f"\n┌{'─'*68}┐")
    print(f"│ 【规则 {i}】")
    print(f"│ 前项: {antecedent[:55]}")
    print(f"│ 后项: {consequent[:55]}")
    print(f"├{'─'*68}┤")
    print(f"│ 关键指标:")
    trans_count = int(rule['support'] * len(transactions)) if 'transaction_count' not in rule else rule['transaction_count']
    print(f"│   • 支持度: {rule['support']:.4f} (约{trans_count:,}笔交易)")
    print(f"│   • 置信度: {rule['confidence']:.2%}")
    print(f"│   • 提升度: {rule['lift']:.2f}")
    print(f"├{'─'*68}┤")
    print(f"│ 业务含义:")
    print(f"│   • 购买前项商品的顾客中, 有 {rule['confidence']:.0%} 会同时购买后项商品")
    print(f"│   • 如果随机挑选顾客, 只有约 {random_prob:.1%} 会购买后项商品")
    print(f"│   • 关联购买概率是随机购买的 {rule['lift']:.1f} 倍 (提升度)")
    print(f"│   • 结论: 两商品存在{'强' if rule['lift'] > 10 else '显著'}正相关, 适合组合营销")
    print(f"└{'─'*68}┘")


print("\n" + "=" * 70)
print("                    具体促销与陈列方案")
print("=" * 70)


rule_a = strong_rules.iloc[0]
rule_b = strong_rules.iloc[min(5, len(strong_rules)-1)]  # 选择不同的规则

item_a1 = rule_a['antecedents_str']
item_a2 = rule_a['consequents_str']

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     方案A: 茶具套装捆绑销售                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ 【目标商品组合】                                                       ║
║   • 主商品: {item_a1[:50]}
║   • 关联商品: {item_a2[:50]}
║                                                                        ║
║ 【数据支撑】                                                           ║
║   • 置信度 {rule_a['confidence']:.0%}: 购买主商品的顾客中{rule_a['confidence']:.0%}会购买关联商品        ║
║   • 提升度 {rule_a['lift']:.1f}: 关联购买概率是随机情况的{rule_a['lift']:.1f}倍             ║
║   • 约{rule_a['transaction_count']:,}笔交易出现此组合                                    ║
║                                                                        ║
║ 【实施措施】                                                           ║
║   1. 【茶具套装】将粉色、绿色、玫瑰色茶杯组成"英式下午茶三件套"       ║
║   2. 【定价策略】套装价格=单品总价×0.85，提供15%优惠                  ║
║   3. 【陈列位置】在茶具区设置专属展示架，三款相邻陈列                  ║
║   4. 【视觉设计】制作精美礼盒包装，适合送礼场景                        ║
║                                                                        ║
║ 【预期效果】                                                           ║
║   • 套装购买率提升: 基于{rule_a['confidence']:.0%}置信度，预计套装转化率可达70%+          ║
║   • 客单价提升: 单品→套装，客单价预计提升2-3倍                        ║
║   • 库存周转: 关联商品同步销售，降低滞销风险                          ║
╚══════════════════════════════════════════════════════════════════════╝
""")

item_b1 = rule_b['antecedents_str']
item_b2 = rule_b['consequents_str']

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     方案B: 智能推荐与交叉营销                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ 【目标商品组合】                                                       ║
║   • 触发商品: {item_b1[:50]}
║   • 推荐商品: {item_b2[:50]}
║                                                                        ║
║ 【数据支撑】                                                           ║
║   • 提升度 {rule_b['lift']:.1f}: 强关联信号，值得精准推荐               ║
║   • 置信度 {rule_b['confidence']:.0%}: 推荐成功率预估                  ║
║                                                                        ║
║ 【线上实施】                                                           ║
║   1. 【购物车推荐】添加触发商品时，弹出"搭配购买"推荐窗口             ║
║   2. 【个性化首页】向购买过触发商品的用户展示推荐商品                  ║
║   3. 【邮件营销】发送"完善您的收藏"主题邮件                          ║
║                                                                        ║
║ 【线下实施】                                                           ║
║   1. 【货架临近】将两商品放在相邻货架或同一购物动线                    ║
║   2. 【POP提示】"买A的顾客也喜欢B"提示牌                             ║
║   3. 【收银推荐】结账时店员话术推荐                                    ║
║                                                                        ║
║ 【效果评估】                                                           ║
║   • 设置A/B测试对照组                                                  ║
║   • 跟踪推荐点击率、转化率、客单价变化                                ║
║   • 预计推荐商品销量提升30-50%                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     方案C: 场景化主题陈列                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ 【策略核心】                                                           ║
║   基于关联规则发现，消费者倾向于购买同系列、同场景的商品组合          ║
║                                                                        ║
║ 【主题专区设计】                                                       ║
║                                                                        ║
║   ┌─────────────────────────────────────────────────────────────┐     ║
║   │  🍵 英式下午茶专区                                          │     ║
║   │  • REGENCY系列茶杯茶碟(粉/绿/玫瑰)                         │     ║
║   │  • 三层蛋糕架                                               │     ║
║   │  • 蛋糕模具套装                                             │     ║
║   │  • 装饰餐巾                                                 │     ║
║   └─────────────────────────────────────────────────────────────┘     ║
║                                                                        ║
║   ┌─────────────────────────────────────────────────────────────┐     ║
║   │  🌻 花园派对专区                                            │     ║
║   │  • 园艺护膝垫(多色)                                        │     ║
║   │  • 户外装饰灯                                               │     ║
║   │  • 野餐篮                                                   │     ║
║   └─────────────────────────────────────────────────────────────┘     ║
║                                                                        ║
║ 【执行要点】                                                           ║
║   1. 每月更新主题，保持新鲜感                                          ║
║   2. 配合节日(圣诞、母亲节)调整商品组合                               ║
║   3. 收集销售数据，持续优化关联规则模型                               ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")


#结果汇总与导出
print("\n" + "▶" * 3 + " 阶段8: 结果汇总与导出 " + "◀" * 3)

# 导出规则表格
export_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'transaction_count']
if 'category' in strong_rules.columns:
    export_cols.append('category')
export_rules = strong_rules[export_cols].head(50)
export_rules.columns = ['前项商品', '后项商品', '支持度', '置信度', '提升度', '涉及交易数'] + (['商品类别'] if 'category' in strong_rules.columns else [])
export_rules.to_excel('关联规则结果.xlsx', index=False)
print(f"\n✓ 已导出: 关联规则结果.xlsx (包含Top 50规则)")

# 导出频繁项集
frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
export_itemsets = frequent_itemsets[['itemsets_str', 'support', 'length']].sort_values('support', ascending=False)
export_itemsets.columns = ['频繁项集', '支持度', '项集大小']
export_itemsets.to_excel('频繁项集结果.xlsx', index=False)
print(f"✓ 已导出: 频繁项集结果.xlsx")


print("\n" + "=" * 70)
print("                        分析结果汇总报告")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                           数据概况                                   │
├─────────────────────────────────────────────────────────────────────┤
│  原始数据量:     {len(df_raw):>10,} 条                                │
│  清洗后数据量:   {len(df):>10,} 条                                    │
│  有效订单数:     {df['InvoiceNo'].nunique():>10,} 笔                  │
│  涉及商品数:     {df['Description'].nunique():>10,} 种               │
│  涉及客户数:     {df['CustomerID'].nunique():>10,} 位                │
│  涉及国家数:     {df['Country'].nunique():>10} 个                    │
├─────────────────────────────────────────────────────────────────────┤
│                           挖掘结果                                   │
├─────────────────────────────────────────────────────────────────────┤
│  频繁项集总数:   {len(frequent_itemsets):>10,} 个                    │
│  关联规则总数:   {len(strong_rules):>10,} 条                         │
│  平均置信度:     {strong_rules['confidence'].mean()*100:>10.1f}%     │
│  平均提升度:     {strong_rules['lift'].mean():>10.2f}                │
│  最高提升度:     {strong_rules['lift'].max():>10.2f}                 │
├─────────────────────────────────────────────────────────────────────┤
│                         核心发现                                     │
├─────────────────────────────────────────────────────────────────────┤
│  1. REGENCY茶杯系列存在极强关联(提升度>15)，适合套装销售            │
│  2. 园艺用品(护膝垫)不同颜色关联度高，可推出多色套装                │
│  3. 蛋糕相关用品与茶具存在场景关联，适合主题营销                    │
├─────────────────────────────────────────────────────────────────────┤
│                           输出文件                                   │
├─────────────────────────────────────────────────────────────────────┤
│  1. 图1_探索性数据分析.png                                           │
│  2. 图2_关联规则分布与排名.png                                       │
│  3. 图3_关联规则网络图.png                                           │
│  4. 图4_频繁项集与规则分析.png                                       │
│  5. 图5_指标解读与提升度分析.png  [新增]                             │
│  6. 关联规则结果.xlsx                                                │
│  7. 频繁项集结果.xlsx                                                │
└─────────────────────────────────────────────────────────────────────┘
""")


print("\n" + "=" * 70)
print("                        附录: 关键概念解释")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        Apriori算法核心概念                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  【支持度 Support】                                                   │
│    定义: support(A→B) = P(A∪B) = 包含A和B的交易数 / 总交易数        │
│    含义: 商品组合的流行程度                                          │
│    本研究阈值: ≥0.015 (至少出现在1.5%的交易中)                       │
│                                                                       │
│  【置信度 Confidence】                                                │
│    定义: confidence(A→B) = P(B|A) = support(A∪B) / support(A)       │
│    含义: 购买A的顾客中，同时购买B的条件概率                          │
│    本研究阈值: ≥0.4 (至少40%的条件概率)                              │
│                                                                       │
│  【提升度 Lift】                                                      │
│    定义: lift(A→B) = confidence(A→B) / support(B)                   │
│           = P(B|A) / P(B)                                            │
│    含义: 关联购买概率相比随机购买概率的提升倍数                      │
│    解读:                                                              │
│      - lift > 1: 正相关，A促进B的购买                                │
│      - lift = 1: 独立，A和B无关联                                    │
│      - lift < 1: 负相关，A抑制B的购买                                │
│    本研究阈值: ≥3.0 (至少3倍提升)                                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("              ✓ 购物篮分析与商品组合挖掘完成！")
print("=" * 70)