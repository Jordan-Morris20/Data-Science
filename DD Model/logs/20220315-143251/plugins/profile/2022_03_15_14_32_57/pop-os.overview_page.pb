?&$	ׂ}?G}@??ӜQ0???D?+?	@!???H??@$	H?sQn@f%5??????h(??!F??$?@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?????&@Uj?@+0@A???q?@YP÷?n???r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1>w??׉@<g?G@A4e??5@Yyͫ:???r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?D?+?	@??>?Q??A Q???@Y?X32?]??r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1#???S	@.S??i??A?K??T @Y?3????r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??ht@?=@?????A?TQ??@Y???????r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????@w?EP??A????J@Yb??U???r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1.????
@????߽??A???Y@Y:vP????r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	&S?r@?|#?g???A??=?>? @YVdt@???r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
??ʦ?@;?a @A6??g?@Y?46<??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???H??@??AA)?@A?s?r@Y2=a????r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???}o@?ݮ??H??Ag??j+v@YCV??#??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1g???p\@]??J??A<???W@Y>?>tA}??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??????@?w???-??A?Cl?pr@Yz?c??T??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?`<#@`?o`r#??A3???Vc@Y???p??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?<,Ԛ?@?e??tg??A???9???Y?w?Go???r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?Z(?|@h?,{??A?|?F?	@Y?5??
??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1t	???!@?ƈD?@A}?b?:?@Y?:U?g$??r	train 517*	????V?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?7?nz??!?;??D@)?j?Q??16?]ȡcA@:Preprocessing2T
Iterator::Root::ParallelMapV2k?	?iz??!>?Y??J2@)k?	?iz??1>?Y??J2@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??Z(???!?}Xa'@)??Z(???1?}Xa'@:Preprocessing2E
Iterator::Roots?9>Z??!??Սb8<@)?f?|?|??1?&?XQ?#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?^b,ӯ??!?????4@)f?2?}???1K=????"@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?ʢ?????!e??F??@)?ʢ?????1e??F??@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip* ?3h@!^??\??Q@)P??????1?????@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap9?Վ???!?¡u=8@){JΉ=???1??1(?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 34.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?!r?8'@I?n?:?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	B?Ǯ???????/?????>?Q??!??AA)?@	!       "	!       *	!       2$	??%@,.l??m?????9???!4e??5@:	!       B	!       J$	?X:$???gj???z?c??T??!CV??#??R	!       Z$	?X:$???gj???z?c??T??!CV??#??b	!       JCPU_ONLYY?!r?8'@b q?n?:?X@Y      Y@q]_?m`??"?
both?Your program is POTENTIALLY input-bound because 34.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 