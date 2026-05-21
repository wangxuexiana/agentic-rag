"""
导入节点集合。

节点职责按执行顺序划分如下：
- 入口与路由：`node_entry`
- 文档预处理：`node_pdf_to_md`、`node_md_img`
- 内容加工：`node_document_split`、`node_item_name_recognition`
- 检索准备：`node_bge_embedding`、`node_import_milvus`
"""
