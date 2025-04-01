import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BWStockModel:
    def __init__(self):
        # 初始化因子权重
        self.factor_weights = {
            'macro': 0.20,    # 宏观因素
            'industry': 0.20, # 行业因素
            'quality': 0.30,  # 公司质量
            'valuation': 0.15,# 估值
            'technical': 0.15 # 技术面
        }
        
        # 买卖阈值
        self.buy_threshold = 70
        self.sell_threshold = 30
        
    def get_macro_data(self):
        """获取宏观经济数据"""
        try:
            # 中国GDP增长率
            gdp_df = ak.macro_china_gdp()
            
            # 处理季度数据
            gdp_df['季度'] = gdp_df['季度'].apply(
                lambda x: f"{x.split('年')[0]}-Q{x.split('第')[1][0]}")
            gdp_df['季度'] = pd.to_datetime(gdp_df['季度'])
            gdp_df.set_index('季度', inplace=True)
            
            # CPI数据
            cpi_df = ak.macro_china_cpi()
            print("原始CPI数据样例:", cpi_df.head())  # 调试用
            
            # 转换中文日期格式 "2025年02月份" → "2025-02"
            cpi_df['月份'] = cpi_df['月份'].str.replace('年', '-').str.replace('月份', '')
            cpi_df['月份'] = pd.to_datetime(cpi_df['月份'], errors='coerce')
            cpi_df = cpi_df.dropna(subset=['月份'])  # 移除无效日期
            cpi_df.set_index('月份', inplace=True)
            
            # 利率数据
            rate_df = ak.macro_china_shibor_all()
            rate_df['日期'] = pd.to_datetime(rate_df['日期'])
            rate_df.set_index('日期', inplace=True)
            
            return {
                'gdp': gdp_df['国内生产总值-绝对值'].pct_change(4).dropna(),
                'cpi': cpi_df['全国-累计'],
                'rate': rate_df['1Y-涨跌幅']
            }
            
        except Exception as e:
            print(f"获取宏观数据时出错: {e}")
            return {
                'gdp': pd.Series(),
                'cpi': pd.Series(),
                'rate': pd.Series()
            }
        
    def get_industry_data(self, industry_code):
        """获取行业数据"""
        # 行业指数数据
        industry_df = ak.stock_board_industry_hist_em(symbol=industry_code)
        industry_df['日期'] = pd.to_datetime(industry_df['日期'])
        industry_df.set_index('日期', inplace=True)
        return industry_df
    
    def get_stock_data(self, stock_code):
        """获取个股数据"""
        # 获取个股历史数据
        stock_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily")
        stock_df['日期'] = pd.to_datetime(stock_df['日期'])
        stock_df.set_index('日期', inplace=True)
        
        # 计算技术指标
        stock_df['50_MA'] = stock_df['收盘'].rolling(50).mean()
        stock_df['200_MA'] = stock_df['收盘'].rolling(200).mean()
        stock_df['RSI'] = self.calculate_rsi(stock_df['收盘'])
        stock_df['MACD'], stock_df['Signal'] = self.calculate_macd(stock_df['收盘'])
        
        # 获取财务数据
        finance_df = ak.stock_financial_report_sina(stock=stock_code, symbol="现金流量表")
        return stock_df, finance_df
    
    def calculate_rsi(self, prices, window=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """计算MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def evaluate_macro(self, macro_data):
        """评估宏观因素"""
        # 简单评分模型
        gdp_score = 80 if macro_data['gdp'][-1] > 0.05 else 60
        cpi_score = 70 if 1 < macro_data['cpi'][-1] < 3 else 50
        rate_score = 80 if macro_data['rate'][-1] < 3 else 60
        return (gdp_score * 0.4 + cpi_score * 0.3 + rate_score * 0.3)
    
    def evaluate_industry(self, industry_data):
        """评估行业因素"""
        returns = industry_data['收盘'].pct_change(20)[-1]
        volatility = industry_data['收盘'].pct_change().std() * np.sqrt(252)
        
        if returns > 0.05:
            return 80
        elif returns > 0:
            return 70
        elif volatility < 0.2:
            return 60
        else:
            return 50
    
    def evaluate_quality(self, finance_data):
        """评估公司质量"""
        # 简化处理 - 实际应用中应从财务数据提取真实指标
        return np.random.randint(60, 90)  # 模拟60-90分的质量评分
    
    def evaluate_valuation(self, stock_data):
        """评估估值"""
        pe_ratio = stock_data['收盘'][-1] / 2  # 模拟PE计算
        if pe_ratio < 15:
            return 80
        elif pe_ratio < 25:
            return 70
        else:
            return 50
    
    def evaluate_technical(self, stock_data):
        """评估技术面"""
        score = 0
        
        # 均线系统
        if stock_data['50_MA'][-1] > stock_data['200_MA'][-1]:
            score += 40
        
        # RSI
        if stock_data['RSI'][-1] < 30:
            score += 30
        elif stock_data['RSI'][-1] < 70:
            score += 20
        else:
            score -= 10
            
        # MACD
        if stock_data['MACD'][-1] > stock_data['Signal'][-1]:
            score += 30
            
        return score
    
    def generate_signals(self, stock_code, industry_code):
        """生成交易信号"""
        # 获取数据
        macro_data = self.get_macro_data()
        industry_data = self.get_industry_data(industry_code)
        stock_data, finance_data = self.get_stock_data(stock_code)
        
        # 因子评估
        macro_score = self.evaluate_macro(macro_data)
        industry_score = self.evaluate_industry(industry_data)
        quality_score = self.evaluate_quality(finance_data)
        valuation_score = self.evaluate_valuation(stock_data)
        technical_score = self.evaluate_technical(stock_data)
        
        # 综合评分
        composite_score = (
            macro_score * self.factor_weights['macro'] +
            industry_score * self.factor_weights['industry'] +
            quality_score * self.factor_weights['quality'] +
            valuation_score * self.factor_weights['valuation'] +
            technical_score * self.factor_weights['technical']
        )
        
        # 生成信号
        signal = '买入' if composite_score >= self.buy_threshold else (
            '卖出' if composite_score <= self.sell_threshold else '持有')
        
        # 可视化
        self.visualize_results(stock_data, composite_score, signal, stock_code)
        
        return {
            '综合评分': composite_score,
            '信号': signal,
            '宏观评分': macro_score,
            '行业评分': industry_score,
            '质量评分': quality_score,
            '估值得分': valuation_score,
            '技术评分': technical_score
        }
    
    def visualize_results(self, stock_data, score, signal, stock_code):
        """可视化结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 价格和均线
        ax1.plot(stock_data.index, stock_data['收盘'], label='价格', color='black')
        ax1.plot(stock_data.index, stock_data['50_MA'], label='50日均线', color='blue')
        ax1.plot(stock_data.index, stock_data['200_MA'], label='200日均线', color='red')
        ax1.set_title(f'{stock_code} - QS个股策略分析 [信号: {signal}]', fontsize=14)
        ax1.legend()
        ax1.grid(True)
        
        # 综合评分
        ax2.axhline(self.buy_threshold, color='green', linestyle='--', label='买入阈值')
        ax2.axhline(self.sell_threshold, color='red', linestyle='--', label='卖出阈值')
        ax2.bar(stock_data.index[-1], score, color='blue', width=5)
        ax2.set_ylim(0, 100)
        ax2.set_title('综合评分')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    model = BWStockModel()
    
    # 输入股票代码和所属行业代码
    stock_code = "300059"  # 东方财富
    industry_code = "证券"  # 行业板块
    
    result = model.generate_signals(stock_code, industry_code)
    print("\n分析结果:")
    for k, v in result.items():
        print(f"{k}: {v}")