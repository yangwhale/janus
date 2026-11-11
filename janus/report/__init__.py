"""渲染：把 results.json 摊开给人看，不重算指标、不改判定。

- ``html``：``janus run --html`` 的 HTML 报告（四级门槛总表、曲线图、逐张量图、全部检查明细）；
- ``tables``：对齐实践 README 里的逐级大表（从各结果文件取数，标记之间可重复生成）；
- ``figures``：逐级大表配的图（L2 逐层、ULP 分桶、路由分差）；
- ``markdown``：``janus run`` 每次都写的 ``report.md``。

画图要 matplotlib（``pip install -e '.[report]'``）。
"""
