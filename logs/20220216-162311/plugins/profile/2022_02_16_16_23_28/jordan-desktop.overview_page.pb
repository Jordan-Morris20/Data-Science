?-$	?Ū8????O3?_?#???'??D??!t`9B???$	?|??'@綸?C-????g?>-@!9h??e%@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???????7n1?74??1h?,{??A a??*??Ix??qo??Y????r	train 197"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??]ؚ?????֥F??1(?$?????A?J?4??I?*?w????Y_?????r	train 198"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCh$B#ظ???@?ش??1?쟧???A??-$`??I?^D?1u??YZ-??DJ??r	train 199"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????????̔????1׆?q???A??u????Im?i?*???Y?VBwI???r	train 200"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?M????::?Fv??1?/??C??A??$Ί???I???-?v??YWZF?=???r	train 201"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?'??D?????U???1???2??A?p=
ף??I???T???Y??qn???r	train 202"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC$?&ݖ????Dg?E(??1?P?%????A??۞ ???I%??1 ???Y?R\U?]??r	train 203"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC	ٵ?ݒ????0?????1??????A.??'Hl??IP?sג??YDܜJ???r	train 204"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC
???0??R)v4???1??E??\??A??R??q??IX??0_???Y??q?d???r	train 205"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCmu9% ????n?|?b??1?R?r/0??A ?g?????IJa??L???Yw?n??\??r	train 206"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?,?????S[? ??1?%??:???A???:U???I9{g?U???Y??z?p̲?r	train 207"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?G?`???N~?N?Z??1yGsd???A?o*Ral??I??5?????Y????%:??r	train 208"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCn?HJzX????fc%???1?9????A&??4??I?-$`t??Y??u??Ź?r	train 209"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?]??a???ZH??????1??_ ??A#,*?t???I?:?z???Y?F?@??r	train 210"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCt`9B???@7n1??1???9d??A??ʦ\??IP??n??Yb??c???r	train 211"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?)X?l??????_????1?Y?rL??AI??&??I`r??Z???Y?8b->??r	train 212"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?;??~????? @????1???o?4??A-wf??\??I?Un2???Yh$B#ظ??r	train 213"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC? 4J?>????:????1I??Z????Aq ?????I?4?\????Y?S???r	train 214"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?$??????v?>X??1?U?3???AoI?????INd?????YG??t???r	train 215*	??(\?e?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatN^????!?b9?K?A@)l??3?I??1.E
?L@@:Preprocessing2T
Iterator::Root::ParallelMapV2?0?*??!k?<4?3@)?0?*??1k?<4?3@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::TensorSliced"??<??!?P3??/@)d"??<??1?P3??/@:Preprocessing2E
Iterator::Root?ꫫ???!??#O??>@)?Y?$?9??1??5?='@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??v???!*??0:@)??5????1????}?$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?׹i3N??!?7?YQQ@)辜ٮЯ?1??Yy?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*!??	L???!?GC?<?@)!??	L???1?GC?<?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?46.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9????B@I???PfDS@Q??[I?0@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	-$i%?[????8K{?????U???!@7n1??	!       "$	,??????Q?"???쟧???!???9d??*	!       2$	?ȋX????>0?M5أ??p=
ף??!??ʦ\??:$	(@)i?J????q?_Q??x??qo??!?Un2???B	!       J$	1??n??4b?Q?o??_?????!?8b->??R	!       Z$	1??n??4b?Q?o??_?????!?8b->??b	!       JGPUY????B@b q???PfDS@y??[I?0@?":
sequential_3/dense_10/MatMulMatMul?ѣ?????!?ѣ?????0"9
sequential_3/dense_9/MatMulMatMul,?rom???!)?3???0":
sequential_3/dense_11/MatMulMatMul]??E?d??!??=3??0"O
1gradient_tape/sequential_3/dense_10/MatMul/MatMulMatMul?{ ?G???!??	Ok??0"O
1gradient_tape/sequential_3/dense_11/MatMul/MatMulMatMul??F?.???!???ޔ`??0"O
3gradient_tape/sequential_3/dense_11/MatMul/MatMul_1MatMul?hq1G ??!???ĝD??"O
3gradient_tape/sequential_3/dense_10/MatMul/MatMul_1MatMul;w5?	??!Vc?E?r??"=
 RMSprop/RMSprop/update_4/truedivRealDiv?-??j??!?B콘9??"7
RMSprop/RMSprop/update_5/addAddV2=
JM?c??!,???????"5
RMSprop/RMSprop/update_5/mulMul "JH???!-cW????Q      Y@Y??k{ ?)@a????_?U@q?M?ؑ?$@y???5???"?
both?Your program is MODERATELY input-bound because 6.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?46.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t18.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?10.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 