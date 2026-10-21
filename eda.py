import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    df = pd.read_csv('data/data_clean.csv')
    
    # Создаем папку для отчетов, если ее нет
    os.makedirs('reports', exist_ok=True)
    
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # 1. Корреляционная матрица (Pearson)
    print("Строим тепловую карту корреляций...")
    plt.figure(figsize=(12, 10))
    num_df = df.select_dtypes(include='number')
    corr = num_df.corr()
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True)
    plt.title('Матрица корреляций признаков', fontsize=16)
    plt.tight_layout()
    plt.savefig('reports/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Boxplot: распределение прибыльности по классам риска
    print("Строим boxplot прибыльности...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='risk_class', y='profitability', data=df, palette='Set2')
    plt.title('Распределение прибыльности по классам риска', fontsize=16)
    plt.xlabel('Класс риска (0-Низкий, 1-Средний, 2-Высокий)', fontsize=12)
    plt.ylabel('Прибыльность (%)', fontsize=12)
    plt.savefig('reports/boxplot_profitability.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Гистограммы распределения ключевых признаков
    print("Строим гистограммы...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Распределение ключевых показателей', fontsize=16)
    
    sns.histplot(df['project_cost'], kde=True, ax=axes[0, 0], color='skyblue', bins=30)
    axes[0, 0].set_title('Стоимость проекта', fontsize=12)
    
    sns.histplot(df['payback_period'], kde=True, ax=axes[0, 1], color='salmon', bins=30)
    axes[0, 1].set_title('Срок окупаемости', fontsize=12)
    
    sns.histplot(df['risk_density'], kde=True, ax=axes[1, 0], color='lightgreen', bins=30)
    axes[1, 0].set_title('Плотность риска', fontsize=12)
    
    sns.histplot(df['capability_score'], kde=True, ax=axes[1, 1], color='plum', bins=30)
    axes[1, 1].set_title('Оценка возможностей', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('reports/histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Countplot: дисбаланс классов
    print("Строим график дисбаланса классов...")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='risk_class', data=df, palette='viridis')
    plt.title('Распределение классов риска в выборке', fontsize=16)
    plt.xlabel('Класс риска', fontsize=12)
    plt.ylabel('Количество проектов', fontsize=12)
    # Подписываем проценты над столбцами
    total = len(df)
    for p in plt.gca().patches:
        height = p.get_height()
        plt.gca().text(p.get_x() + p.get_width()/2., height + 5,
                       f'{height/total:.1%}', ha="center")
    plt.savefig('reports/class_imbalance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("EDA завершено. Все графики сохранены в папку reports/")

if __name__ == '__main__':
    run_eda()