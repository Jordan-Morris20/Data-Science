?&$	???&??@??U?? ????t?(@!?Q?ګ@$	}?䗰?@X????????g(???!>??՞?"@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1&S??@&?B????Aŭ???@Y??E`?o??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1%]3?f?@?ѩ+????A???@Yz?????r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1N
?gz	@???????A?????@Y?????ڴ?r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?>?0@C??u??A?\??m
@Y???bc^??r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1(E+?s@P?s'????A[#?qp? @Y'?y?3??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?B?O??@EKOˏ??A?'?X? @Ys?V{???r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?f?@?@??L??A?xZ~?*
@Y??h?'???r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	?[???@
?_??M??A?L?T	@Y?	.V?`??r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
V??{?
@a?xwd???Aw?h?hS@Y?e6\??r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1RE?*k{@f??S9-??A?ra*@YN??1?M??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?g???@2??????A1е/ @Y?2???r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?Q?ګ@vP??7	@A????<L@YQ??????r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?֊6?9@N??1????A׽?	?@Y??jGq???r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??t?(@??&?????A??¼?9 @Y?????r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???ɍ2@p>u?Rz??A?>#k@Yf?y?????r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?&PĲ@??????A??3??@Y5s?????r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1???_wz@??h:;???A????c5 @Y崧??س?r	train 517*	/?$?L?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeati㈵????!??? @pE@)q?i݆??1ʶ0?B@:Preprocessing2T
Iterator::Root::ParallelMapV2m?kA???!???G?2@)m?kA???1???G?2@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?*5{???!X?_'@)?*5{???1X?_'@:Preprocessing2E
Iterator::Root?9????!??$?"=@)S?'??Z??1??4??0%@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip;?O??. @!?9ԶY?Q@)?Ƅ?K???1@z??	b@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??]?p??!?m?@)??]?p??1?m?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2*A*??!2??? ?/@)\??AA)??1???ၱ@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMaps???e??!???-d$4@)???ʅ??1???N?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 30.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?,{?@@I?&?!?%X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?!#S?r?????^jw????&?????!vP??7	@	!       "	!       *	!       2$	/ɴ??@|y???????c5 @!??3??@:	!       B	!       J$	1J,??S???}??Vó??????!z?????R	!       Z$	1J,??S???}??Vó??????!z?????b	!       JCPU_ONLYY?,{?@@b q?&?!?%X@Y      Y@qj?QN???"?
both?Your program is POTENTIALLY input-bound because 30.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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