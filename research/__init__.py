"""研究层：PIT 股票池、字段补齐、数据清洗与池层验证。

与生产层的边界
--------------
``app.py`` / ``analyzer.py`` / ``market_data.py`` 保持零第三方依赖、只服务单标的
实时分析；本包允许依赖 ``baostock``（研究用数据源，含 PIT 成分与后复权日线），
但**只在抓取入口**``research.bs_source``里 import，其余模块纯标准库，因此
测试与回测可以在没装 baostock 的机器上跑。
"""
