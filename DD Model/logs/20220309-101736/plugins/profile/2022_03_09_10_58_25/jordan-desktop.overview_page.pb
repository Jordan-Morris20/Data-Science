?.$	?p???@????????C????@!?#??ŋ@$	A?޸?4@???&??@?vh @!???/?)@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?`⏢?@g(?x????1Gx$(??A'?_???I^???j???Y??D.8???r	train 500"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCS??i@??Rx????1??9̗??A??o?DI??I?{??????Y???u?|??r	train 501"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?d?`T?@?'?$????1/?u?;O??A??????I???(t??Y??b?????r	train 502"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCdw??+@Pj?L??1?/K;5???AʉvR~??IW{????Y,I???p??r	train 503"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?#??ŋ@ZF?=?S??1=)?? ??A??3?ތ??I|?ڥ'@Y[?7?qÿ?r	train 504"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?dV?p@A-?>??1?:?vٯ??A???6????I?4?B??Ywg????r	train 505"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCU?W?&@J??{d???1?䠄???A3?z????I?K?1?=??YYL?Q???r	train 506"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC		?P??@.??????1?B?????Aܠ?[;Q??I???8???YV??#)??r	train 507"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC
?'?8G@s۾G????1w1?t????AJ???`??I*A*?N??Y??W?????r	train 508"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?#+??@'3?Vz???1?X6sHj??Ao+?6+??IԻx?n???Y???G????r	train 509"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??$"?@??5\???1????4)??Aq:?V???I;??????YP?s'???r	train 510"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCgҦ?y@?6ɏ????1??++MJ??A????????IzȔA??Y?74e???r	train 511"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?C????@??&????1??ZD???A/?ͮ{??I??R{m??Y>#?ƽ?r	train 512"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????H@?V?9?m??1???W;???A??v?$$??ImU???YH???????r	train 513"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCT???@Q?5?U??1?2?,%??AS?
cA??I??׹i???YY???tw??r	train 514"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???-N
@?|x? c??15D?o??A??????I???
D???Y?jQLް?r	train 515"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC;?I?@]??J???1x??#????A1%??e??I?N\?W???YUN{JΉ??r	train 516"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?eo)g@"QhY????1B[Υ????AU?q7????IyX?5????Yi?^`V(??r	train 517"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?)?n?%@C?*q??1?XQ?i??AC9ѮB???I???q?D??Y:???????r	train 518*	P??n?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??/Ie???!???p8?D@)??????1at?\rC@:Preprocessing2T
Iterator::Root::ParallelMapV2???A?p??!???t=z0@)???A?p??1???t=z0@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice8?W??V??!??4??*@)8?W??V??1??4??*@:Preprocessing2E
Iterator::Root*?~????!`F??9?:@)Z?wg???1?;???$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?????!h??1FR@)???!o???14???"@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?ݑ?????!??nL?6@)8J^?c@??1?? Se@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate9)?{?i??!??C???1@)?б?J??1Gƥ?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?c?3?%??!???B?M@)?c?3?%??1???B?M@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?51.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t22.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?????@Ix??s0sT@Q??S??X'@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??Lf£??$?X????.??????!?|x? c??	!       "$	?^ӌ3????
M??????W;???!=)?? ??*	!       2$	?"?˩????Eߪ0???/?ͮ{??!??3?ތ??:$	?SR9B??HD?.?????R{m??!|?ڥ'@B	!       J$	?8?{????ݧ?t????jQLް?!P?s'???R	!       Z$	?8?{????ݧ?t????jQLް?!P?s'???b	!       JGPUY?????@b qx??s0sT@y??S??X'@?"N
0gradient_tape/sequential_2/dense_7/MatMul/MatMulMatMulmi?g???!mi?g???0"9
sequential_2/dense_7/MatMulMatMulR??S?O??!`?	2s??0"N
0gradient_tape/sequential_2/dense_6/MatMul/MatMulMatMul??i{???!3?????0"N
2gradient_tape/sequential_2/dense_7/MatMul/MatMul_1MatMul???
T??!??!????"9
sequential_2/dense_6/MatMulMatMulołW???!?S??]$??0"^
;sequential_2/dropout_2/dropout/random_uniform/RandomUniformRandomUniform08?ݟ??!@?qG??"7
dense_7/kernel/Regularizer/SumSum???7???!?;??Xe??"R
/gradient_tape/binary_crossentropy/DynamicStitchDynamicStitch????P??!Z???F??"N
2gradient_tape/sequential_2/dense_8/MatMul/MatMul_1MatMul????э?!???w=#??"9
 dense_7/kernel/Regularizer/Sum_1Sum??{2?V??!????????Q      Y@Y?¶?-&@a??'?I:V@q?uCn-@yv?;$??"?
both?Your program is MODERATELY input-bound because 6.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?51.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t22.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?14.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 