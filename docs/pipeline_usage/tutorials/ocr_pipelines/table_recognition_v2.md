---
comments: true
---

# 通用表格识别产线v2使用教程

## 1. 通用表格识别产线v2介绍
表格识别是一种自动从文档或图像中识别和提取表格内容及其结构的技术，广泛应用于数据录入、信息检索和文档分析等领域。通过使用计算机视觉和机器学习算法，表格识别能够将复杂的表格信息转换为可编辑的格式，方便用户进一步处理和分析数据。

通用表格识别产线v2用于解决表格识别任务，对图片中的表格进行识别，并以HTML格式输出。与通用表格识别产线不同，本产线引入了表格分类和表格单元格检测两个模块，并将其与表格结构识别模块串联以完成表格识别任务。基于本产线，可实现对表格的精准预测，使用场景覆盖通用、制造、金融、交通等各个领域。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/01.png">

<b>通用</b><b>表格识别</b><b>产线v2中包含必选的表格结构识别模块、表格分类模块、表格单元格定位模块、文本检测模块和文本识别模块，以及可选的版面区域检测模块、文档图像方向分类模块和文本图像矫正模块</b>。

<b>如果您更注重模型的精度，请选择精度较高的模型；如果您更在意模型的推理速度，请选择推理速度较快的模型；如果您关注模型的存储大小，请选择存储体积较小的模型。</b>


<p><b>表格识别模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>精度（%）</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>SLANet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SLANet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">训练模型</a></td>
<td>59.52</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
<td rowspan="2">SLANet 是百度飞桨视觉团队自研的表格结构识别模型。该模型通过采用CPU 友好型轻量级骨干网络PP-LCNet、高低层特征融合模块CSP-PAN、结构与位置信息对齐的特征解码模块SLA Head，大幅提升了表格结构识别的精度和推理速度。</td>
</tr>
<tr>
<td>SLANet_plus</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/SLANet_plus_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">训练模型</a></td>
<td>63.69</td>
<td>522.536</td>
<td>1845.37</td>
<td>6.9 M</td>
</tr>
</table>
<p><b>注：以上精度指标测量PaddleX 内部自建英文表格识别数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>


<p><b>表格分类模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top1 Acc(%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/CLIP_vit_base_patch16_224_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">训练模型</a></td>
<td>--</td>
<td>--</td>
<td>--</td>
<td>--</td>
</tr>
</table>
<p><b>注：以上精度指标测量自 PaddleX 内部自建表格分类数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p></details>


<p><b>表格单元格检测模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wired_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">--</td>
<td rowspan="2">RT-DETR 是第一个实时的端到端目标检测模型。百度飞桨视觉团队基于 RT-DETR-L 作为基础模型，在自建表格单元格检测数据集上完成预训练，实现了对有线表格、无线表格均有较好性能的表格单元格检测。
</td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0rc0/RT-DETR-L_wireless_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
</tr>
</table>
<p><b>注：以上精度指标测量自 PaddleX 内部自建表格单元格检测数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p></details>


<p><b>文本检测模块模型：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
<td>82.69</td>
<td>83.3501</td>
<td>2434.01</td>
<td>109</td>
<td>PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
<td>77.79</td>
<td>10.6923</td>
<td>120.177</td>
<td>4.7</td>
<td>PP-OCRv4 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>


<p><b>文本识别模块模型：</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
<td>78.20</td>
<td>7.95018</td>
<td>46.7868</td>
<td>10.6 M</td>
<td rowspan="2">PP-OCRv4是百度飞桨视觉团队自研的文本识别模型PP-OCRv3的下一个版本，通过引入数据增强方案、GTC-NRTR指导分支等策略，在模型推理速度不变的情况下，进一步提升了文本识别精度。该模型提供了服务端（server）和移动端（mobile）两个不同版本，来满足不同场景下的工业需求。</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
<td>79.20</td>
<td>7.19439</td>
<td>140.179</td>
<td>71.2 M</td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td>
<td>68.81</td>
<td>8.36801</td>
<td>165.706</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 是一种由复旦大学视觉与学习实验室（FVL）的OpenOCR团队研发的服务端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，A榜端到端识别精度相比PP-OCRv4提升6%。
</td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td>
<td>65.07</td>
<td>10.5047</td>
<td>51.5647</td>
<td>22.1 M</td>
<td rowspan="1">    RepSVTR 文本识别模型是一种基于SVTRv2 的移动端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。</td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜。 所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>


<p><b>版面区域检测模块模型（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
<td>86.8</td>
<td>13.0</td>
<td>91.3</td>
<td>7.4</td>
<td>基于PicoDet-1x在PubLayNet数据集训练的高效率版面区域定位模型，可定位包含文字、标题、表格、图片以及列表这5类区域</td>
</tr>
<tr>
<td>PicoDet_layout_1x_table</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
<td>95.7</td>
<td>12.623</td>
<td>90.8934</td>
<td>7.4 M</td>
<td>基于PicoDet-1x在自建数据集训练的高效率版面区域定位模型，可定位包含表格这1类区域</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>87.1</td>
<td>13.5</td>
<td>45.8</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>70.3</td>
<td>13.6</td>
<td>46.2</td>
<td>4.8</td>
<td>基于PicoDet-S轻量模型在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>89.3</td>
<td>15.7</td>
<td>159.8</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>79.9</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td>基于PicoDet-L在中英文论文、杂志和研报等场景上自建数据集训练的高效率版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
<td>95.9</td>
<td>114.6</td>
<td>3832.6</td>
<td>470.1</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型，包含3个类别：表格，图像和印章</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
<td>92.6</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td>基于RT-DETR-H在中英文论文、杂志和研报等场景上自建数据集训练的高精度版面区域定位模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域分析数据集，包含中英文论文、杂志和研报等常见的 1w 张文档类型图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>


<p><b>文本图像矫正模块模型（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>MS-SSIM （%）</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td>
<td>54.40</td>
<td>30.3 M</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
<p><b>模型的精度指标测量自 <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet benchmark</a>。</b></p>


<p><b>文档图像方向分类模块模型（可选）：</b></p>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小（M)</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
<td>99.06</td>
<td>3.84845</td>
<td>9.23735</td>
<td>7</td>
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为 8，精度类型为 FP32。</b></p>


## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在本地使用命令行或 Python 体验通用表格识别产线v2的效果。

### 2.1 在线体验
暂不支持在线体验。

### 2.2 本地体验
在本地使用通用表格识别产线v2前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验表格识别产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline table_recognition_v2 \
        --input table_recognition.jpg \
        --save_path ./output \
        --device gpu:0
```

相关的参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的参数说明。

运行后，会将结果打印到终端上，结果如下：

```bash
{'res': {'input_path': 'table_recognition.jpg', 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_ocr_model': True}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 0, 'label': 'Table', 'score': 0.9922188520431519, 'coordinate': [3.0127392, 0.14648987, 547.5102, 127.72023]}]}, 'overall_ocr_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'dt_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 'text_rec_score_thresh': 0, 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上', '没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': [0.9951260685920715, 0.9943379759788513, 0.9968608021736145, 0.9978817105293274, 0.9985721111297607, 0.9616036415100098, 0.9977153539657593, 0.987593948841095, 0.9906861186027527, 0.9959743618965149, 0.9970152378082275, 0.9977849721908569, 0.9984450936317444], 'rec_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'rec_boxes': array([[234,   6, 316,  25],
       [ 38,  39,  73,  57],
       [122,  32, 201,  58],
       [227,  34, 346,  57],
       [351,  34, 391,  58],
       [417,  35, 534,  58],
       [ 34,  70,  78,  90],
       [287,  70, 328,  90],
       [454,  69, 496,  90],
       [ 17, 101,  95, 124],
       [144, 101, 178, 122],
       [278, 101, 338, 124],
       [448, 101, 503, 121]], dtype=int16)}, 'table_res_list': [{'cell_box_list': [array([3.18822289e+00, 1.46489874e-01, 5.46996138e+02, 3.08782365e+01]), array([  3.21032453,  31.1510637 , 110.20750237,  65.14108063]), array([110.18174553,  31.13076188, 213.00813103,  65.02860047]), array([212.96108818,  31.09959008, 404.19618034,  64.99535157]), array([404.08112907,  31.18304802, 547.00864983,  65.0847223 ]), array([  3.21772957,  65.0738733 , 110.33685875,  96.07921387]), array([110.23703575,  65.02486207, 213.08839226,  96.01378419]), array([213.06095695,  64.96230103, 404.28425407,  95.97141816]), array([404.23704338,  65.04879548, 547.01273918,  96.03654267]), array([  3.22793937,  96.08334137, 110.38572502, 127.08698823]), array([110.40586662,  96.10539795, 213.19943047, 127.07002045]), array([213.12627983,  96.0539148 , 404.42686272, 127.02842499]), array([404.33042717,  96.07251526, 547.01273918, 126.45088746])], 'pred_html': '<html><body><table><tr><td colspan="4">CRuncover</td></tr><tr><td>Dres</td><td>连续工作3</td><td>取出来放在网上 没想</td><td>江、整江等八大</td></tr><tr><td>Abstr</td><td></td><td>rSrivi</td><td>$709.</td></tr><tr><td>cludingGiv</td><td>2.72</td><td>Ingcubic</td><td>$744.78</td></tr></table></body></html>', 'table_ocr_pred': {'rec_polys': [array([[234,   6],
       [316,   6],
       [316,  25],
       [234,  25]], dtype=int16), array([[38, 39],
       [73, 39],
       [73, 57],
       [38, 57]], dtype=int16), array([[122,  32],
       [201,  32],
       [201,  58],
       [122,  58]], dtype=int16), array([[227,  34],
       [346,  34],
       [346,  57],
       [227,  57]], dtype=int16), array([[351,  34],
       [391,  34],
       [391,  58],
       [351,  58]], dtype=int16), array([[417,  35],
       [534,  35],
       [534,  58],
       [417,  58]], dtype=int16), array([[34, 70],
       [78, 70],
       [78, 90],
       [34, 90]], dtype=int16), array([[287,  70],
       [328,  70],
       [328,  90],
       [287,  90]], dtype=int16), array([[454,  69],
       [496,  69],
       [496,  90],
       [454,  90]], dtype=int16), array([[ 17, 101],
       [ 95, 101],
       [ 95, 124],
       [ 17, 124]], dtype=int16), array([[144, 101],
       [178, 101],
       [178, 122],
       [144, 122]], dtype=int16), array([[278, 101],
       [338, 101],
       [338, 124],
       [278, 124]], dtype=int16), array([[448, 101],
       [503, 101],
       [503, 121],
       [448, 121]], dtype=int16)], 'rec_texts': ['CRuncover', 'Dres', '连续工作3', '取出来放在网上', '没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_scores': [0.9951260685920715, 0.9943379759788513, 0.9968608021736145, 0.9978817105293274, 0.9985721111297607, 0.9616036415100098, 0.9977153539657593, 0.987593948841095, 0.9906861186027527, 0.9959743618965149, 0.9970152378082275, 0.9977849721908569, 0.9984450936317444], 'rec_boxes': [array([234,   6, 316,  25], dtype=int16), array([38, 39, 73, 57], dtype=int16), array([122,  32, 201,  58], dtype=int16), array([227,  34, 346,  57], dtype=int16), array([351,  34, 391,  58], dtype=int16), array([417,  35, 534,  58], dtype=int16), array([34, 70, 78, 90], dtype=int16), array([287,  70, 328,  90], dtype=int16), array([454,  69, 496,  90], dtype=int16), array([ 17, 101,  95, 124], dtype=int16), array([144, 101, 178, 122], dtype=int16), array([278, 101, 338, 124], dtype=int16), array([448, 101, 503, 121], dtype=int16)]}}]}}
```
运行结果参数说明可以参考[2.2.2 Python脚本方式集成](#222-python脚本方式集成)中的结果解释。

可视化结果保存在`save_path`下，其中表格识别的可视化结果如下：
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition_v2/03.png">


### 2.2 Python脚本方式集成
* 上述命令行是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline_name="table_recognition_v2")

output = pipeline.predict(
    input="table_recognition.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print() 
    res.save_to_img("./output/") 
    res.save_to_xlsx("./output/") 
    res.save_to_html("./output/") 
    res.save_to_json("./output/") 
```

在上述 Python 脚本中，执行了如下几个步骤：

（1）通过 `create_pipeline()` 实例化通用表格识别产线v2对象，具体参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备。支持指定GPU具体卡号，如“gpu:0”，其他硬件具体卡号，如“npu:0”，CPU如“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu:0</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>

（2）调用通用表格识别产线v2对象的 `predict()` 方法进行推理预测。该方法将返回一个 `generator`。以下是 `predict()` 方法的参数及其说明：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填</td>
<td><code>Python Var|str|list</code></td>
<td>
<ul>
  <li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
  <li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
  <li><b>List</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线推理设备</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
  <li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
  <li><b>NPU</b>：如 <code>npu:0</code> 表示使用第 1 块 NPU 进行推理；</li>
  <li><b>XPU</b>：如 <code>xpu:0</code> 表示使用第 1 块 XPU 进行推理；</li>
  <li><b>MLU</b>：如 <code>mlu:0</code> 表示使用第 1 块 MLU 进行推理；</li>
  <li><b>DCU</b>：如 <code>dcu:0</code> 表示使用第 1 块 DCU 进行推理；</li>
  <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化时，会优先使用本地的 GPU 0号设备，如果没有，则使用 CPU 设备；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否使用文档方向分类模块</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
  <li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否使用文档扭曲矫正模块</td>
<td><code>bool|None</code></td>
<td>
<ul>
  <li><b>bool</b>：<code>True</code> 或者 <code>False</code>；</li>
  <li><b>None</b>：如果设置为<code>None</code>, 将默认使用产线初始化的该参数值，初始化为<code>True</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>text_det_limit_side_len</code></td>
<td>文本检测的图像边长限制</td>
<td><code>int|None</code></td>
<td>
<ul>
  <li><b>int</b>：大于 <code>0</code> 的任意整数；</li>
  <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>960</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>text_det_limit_type</code></td>
<td>文本检测的图像边长限制类型</td>
<td><code>str|None</code></td>
<td>
<ul>
  <li><b>str</b>：支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code></li>
  <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值，初始化为 <code>max</code>；</li>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>text_det_thresh</code></td>
<td>检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.3</code></td>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>text_det_box_thresh</code></td>
<td>检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.6</code></td>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>2.0</code></td>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留</td>
<td><code>float|None</code></td>
<td>
<ul>
    <li><b>float</b>：大于 <code>0</code> 的任意浮点数
    <li><b>None</b>：如果设置为 <code>None</code>, 将默认使用产线初始化的该参数值 <code>0.0</code>。即不设阈值</td>
</ul>
</ul>
</td>
<td><code>None</code></td>
</tr>

</table>

（3）对预测结果进行处理，每个样本的预测结果均为`dict`类型，且支持打印、保存为图片、保存为`xlsx`文件、保存为`HTML`文件、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>将结果保存为xlsx格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>
<tr>
<td><code>save_to_html()</code></td>
<td>将结果保存为html格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，支持目录或文件路径</td>
<td>无</td>
</tr>


</table>

- 调用`print()` 方法会将结果打印到终端，打印到终端的内容解释如下：

    - `input_path`: `(str)` 待预测图像的输入路径

    - `model_settings`: `(Dict[str, bool])` 配置产线所需的模型参数

        - `use_doc_preprocessor`: `(bool)` 控制是否启用文档预处理子产线
    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` 文档预处理子产线的输出结果。仅当`use_doc_preprocessor=True`时存在
        - `input_path`: `(Union[str, None])` 图像预处理子产线接受的图像路径，当输入为`numpy.ndarray`时，保存为`None`
        - `model_settings`: `(Dict)` 预处理子产线的模型配置参数
            - `use_doc_orientation_classify`: `(bool)` 控制是否启用文档方向分类
            - `use_doc_unwarping`: `(bool)` 控制是否启用文档扭曲矫正
        - `angle`: `(int)` 文档方向分类的预测结果。启用时取值为[0,1,2,3]，分别对应[0°,90°,180°,270°]；未启用时为-1

    - `dt_polys`: `(List[numpy.ndarray])` 文本检测的多边形框列表。每个检测框由4个顶点坐标构成的numpy数组表示，数组shape为(4, 2)，数据类型为int16

    - `dt_scores`: `(List[float])` 文本检测框的置信度列表

    - `text_det_params`: `(Dict[str, Dict[str, int, float]])` 文本检测模块的配置参数
        - `limit_side_len`: `(int)` 图像预处理时的边长限制值
        - `limit_type`: `(str)` 边长限制的处理方式
        - `thresh`: `(float)` 文本像素分类的置信度阈值
        - `box_thresh`: `(float)` 文本检测框的置信度阈值
        - `unclip_ratio`: `(float)` 文本检测框的膨胀系数
        - `text_type`: `(str)` 文本检测的类型，当前固定为"general"

    - `text_rec_score_thresh`: `(float)` 文本识别结果的过滤阈值

    - `rec_texts`: `(List[str])` 文本识别结果列表，仅包含置信度超过`text_rec_score_thresh`的文本

    - `rec_scores`: `(List[float])` 文本识别的置信度列表，已按`text_rec_score_thresh`过滤

    - `rec_polys`: `(List[numpy.ndarray])` 经过置信度过滤的文本检测框列表，格式同`dt_polys`

    - `rec_boxes`: `(numpy.ndarray)` 检测框的矩形边界框数组，shape为(n, 4)，dtype为int16。每一行表示一个矩形框的[x_min, y_min, x_max, y_max]坐标
    ，其中(x_min, y_min)为左上角坐标，(x_max, y_max)为右下角坐标

- 调用`save_to_json()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}.json`，如果指定为文件，则直接保存到该文件中。由于json文件不支持保存numpy数组，因此会将其中的`numpy.array`类型转换为列表形式。
- 调用`save_to_img()` 方法会将可视化结果保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}_ocr_res_img.{your_img_extension}`，如果指定为文件，则直接保存到该文件中。(产线通常包含较多结果图片，不建议直接指定为具体的文件路径，否则多张图会被覆盖，仅保留最后一张图)
- 调用`save_to_html()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}.html`，如果指定为文件，则直接保存到该文件中。在通用表格识别产线v2中，将会把图像中表格的HTML形式写入到指定的html文件中。
- 调用`save_to_xlsx()` 方法会将上述内容保存到指定的`save_path`中，如果指定为目录，则保存的路径为`save_path/{your_img_basename}.xlsx`，如果指定为文件，则直接保存到该文件中。在通用表格识别产线v2中，将会把图像中表格的Excel表格形式写入到指定的xlsx文件中。

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的 <code>json</code> 格式的结果</td>
</tr>
<tr>
<td rowspan = "2"><code>img</code></td>
<td rowspan = "2">获取格式为 <code>dict</code> 的可视化图像</td>
</tr>
</table>

- `json` 属性获取的预测结果为dict类型的数据，相关内容与调用 `save_to_json()` 方法保存的内容一致。
- `img` 属性返回的预测结果是一个字典类型的数据。其中，键分别为 `table_res_img`、`ocr_res_img` 、`layout_res_img` 和 `preprocessed_img`，对应的值是四个 `Image.Image` 对象，按顺序分别为：表格识别结果的可视化图像、OCR 结果的可视化图像、版面区域检测结果的可视化图像、图像预处理的可视化图像。如果没有使用某个子模块，则字典中不包含对应的结果图像。

此外，您可以获取通用表格识别产线v2配置文件，并加载配置文件进行预测。可执行如下命令将结果保存在 `my_path` 中：

```
paddlex --get_pipeline_config table_recognition_v2 --save_path ./my_path
```

若您获取了配置文件，即可对通用表格识别产线v2各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。示例如下：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline_name="./my_path/table_recognition_v2.yaml")

output = pipeline.predict(
    input="table_recognition.jpg",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

for res in output:
    res.print() 
    res.save_to_img("./output/") 
    res.save_to_xlsx("./output/") 
    res.save_to_html("./output/") 
    res.save_to_json("./output/") 

```

<b>注：</b> 配置文件中的参数为产线初始化参数，如果希望更改通用通用表格识别产线v2初始化参数，可以直接修改配置文件中的参数，并加载配置文件进行预测。同时，CLI 预测也支持传入配置文件，`--pipeline` 指定配置文件的路径即可。


## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持多种产线服务化部署方案，详细的产线服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<details><summary>API参考</summary>

<p>对于服务提供的主要操作：</p>
<ul>
<li>HTTP请求方法为POST。</li>
<li>请求体和响应体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>操作结果。</td>
</tr>
</tbody>
</table>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的主要操作如下：</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>定位并识别图中的表格。</p>
<p><code>POST /table-recognition</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>file</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件或PDF文件的URL，或上述类型文件内容的Base64编码结果。对于超过10页的PDF文件，只有前10页的内容会被使用。</td>
<td>是</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code></td>
<td>文件类型。<code>0</code>表示PDF文件，<code>1</code>表示图像文件。若请求体无此属性，则将根据URL推断文件类型。</td>
<td>否</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>tableRecResults</code></td>
<td><code>object</code></td>
<td>表格识别结果。数组长度为1（对于图像输入）或文档页数与10中的较小者（对于PDF输入）。对于PDF输入，数组中的每个元素依次表示PDF文件中每一页的处理结果。</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>输入数据信息。</td>
</tr>
</tbody>
</table>
<p><code>tableRecResults</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>tables</code></td>
<td><code>array</code></td>
<td>表格位置和内容。</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>版面区域检测结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
<tr>
<td><code>ocrImage</code></td>
<td><code>string</code></td>
<td>OCR结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>tables</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>表格位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。</td>
</tr>
<tr>
<td><code>html</code></td>
<td><code>string</code></td>
<td>HTML格式的表格识别结果。</td>
</tr>
</tbody>
</table></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/table-recognition&quot;
file_path = &quot;./demo.jpg&quot;

with open(file_path, &quot;rb&quot;) as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode(&quot;ascii&quot;)

payload = {&quot;file&quot;: file_data, &quot;fileType&quot;: 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
for i, res in enumerate(result[&quot;tableRecResults&quot;]):
    print(&quot;Detected tables:&quot;)
    print(res[&quot;tables&quot;])
    layout_img_path = f&quot;layout_{i}.jpg&quot;
    with open(layout_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;layoutImage&quot;]))
    ocr_img_path = f&quot;ocr_{i}.jpg&quot;
    with open(ocr_img_path, &quot;wb&quot;) as f:
        f.write(base64.b64decode(res[&quot;ocrImage&quot;]))
    print(f&quot;Output images saved at {layout_img_path} and {ocr_img_path}&quot;)
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用表格识别产线v2提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升通用表格识别产线v2的在您的场景中的识别效果。

### 4.1 模型微调
由于通用表格识别产线v2包含若干模块，模型产线的效果如果不及预期，可能来自于其中任何一个模块。您可以对识别效果差的图片进行分析，进而确定是哪个模块存在问题，并参考以下表格中对应的微调教程链接进行模型微调。

<table>
  <thead>
    <tr>
      <th>情形</th>
      <th>微调模块</th>
      <th>微调参考链接</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>表格结分类错误</td>
      <td>表格分类模块</td>
      <td><a href="../../../module_usage/tutorials/ocr_modules/table_classification.md">链接</a></td>
    </tr>
    <tr>
      <td>表格单元格定位错误</td>
      <td>表格单元格检测模块</td>
      <td><a href="../../../module_usage/tutorials/ocr_modules/table_cells_detection.md">链接</a></td>
    </tr>
    <tr>
      <td>表格结构识别错误</td>
      <td>表格结构识别模块</td>
      <td><a href="../../../module_usage/tutorials/ocr_modules/table_structure_recognition.md">链接</a></td>
    </tr>
    <tr>
      <td>未能成功检测到表格所在区域</td>
      <td>版面区域检测模块</td>
      <td><a href="../../../module_usage/tutorials/ocr_modules/layout_detection.md">链接</a></td>
    </tr>
    <tr>
      <td>文本存在漏检</td>
      <td>文本检测模块</td>
      <td><a href="../../../module_usage/tutorials/ocr_modules/text_detection.md">链接</a></td>
    </tr>
    <tr>
      <td>文本内容都不准</td>
      <td>文本识别模块</td>
      <td><a href="../../../module_usage/tutorials/ocr_modules/text_recognition.md">链接</a></td>
    </tr>
    <tr>
      <td>整图旋转矫正不准</td>
      <td>文档图像方向分类模块</td>
      <td><a href="../../../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md">链接</a></td>
    </tr>
    <tr>
      <td>图像扭曲矫正不准</td>
      <td>文本图像矫正模块</td>
      <td>暂不支持微调</td>
    </tr>
  </tbody>
</table>

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```yaml
SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PicoDet_layout_1x_table
    model_dir: null # 替换为微调后的版面区域检测模型权重路径

  TableClassification:
    module_name: table_classification
    model_name: PP-LCNet_x1_0_table_cls
    model_dir: null # 替换为微调后的表格分类模型权重路径

  WiredTableStructureRecognition:
    module_name: table_structure_recognition
    model_name: SLANeXt_wired
    model_dir: null # 替换为微调后的有线表格结构识别模型权重路径
  
  WirelessTableStructureRecognition:
    module_name: table_structure_recognition
    model_name: SLANeXt_wireless
    model_dir: null # 替换为微调后的无线表格结构识别模型权重路径
  
  WiredTableCellsDetection:
    module_name: table_cells_detection
    model_name: RT-DETR-L_wired_table_cell_det
    model_dir: null # 替换为微调后的有线表格单元格检测模型权重路径
  
  WirelessTableCellsDetection:
    module_name: table_cells_detection
    model_name: RT-DETR-L_wireless_table_cell_det
    model_dir: null # 替换为微调后的无线表格单元格检测模型权重路径

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    use_doc_orientation_classify: True
    use_doc_unwarping: True
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null # 替换为微调后的文档图像方向分类模型权重路径

      DocUnwarping:
        module_name: image_unwarping
        model_name: UVDoc
        model_dir: null

  GeneralOCR:
    pipeline_name: OCR
    text_type: general
    use_doc_preprocessor: False
    use_textline_orientation: False
    SubModules:
      TextDetection:
        module_name: text_detection
        model_name: PP-OCRv4_server_det
        model_dir: null # 替换为微调后的文本检测模型权重路径
        limit_side_len: 960
        limit_type: max
        thresh: 0.3
        box_thresh: 0.6
        unclip_ratio: 2.0
        
      TextRecognition:
        module_name: text_recognition
        model_name: PP-OCRv4_server_rec
        model_dir: null # 替换为微调后文本识别的模型权重路径
        batch_size: 1
        score_thresh: 0
```
随后， 参考[2.2 本地体验](#22-本地体验)中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device` 参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用昇腾 NPU 进行 OCR 产线的推理，使用的 Python 命令为：

```bash
paddlex --pipeline table_recognition_v2 \
        --input table_recognition.jpg \
        --save_path ./output \
        --device npu:0
```
若您想在更多种类的硬件上使用通用表格识别产线v2，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。