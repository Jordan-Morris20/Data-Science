?&$	?b????~|????%!???'??!????yN??$	?jV?s@??????=???7P@!^??Կ?@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1%!???'???4S??A?*??	??Y?.ޏ?/??r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1f/?N[??@Qٰ????A$??????Y???_vO??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1=?Е????&N???A????????Y?Đ?Lܢ?r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?ΤMս????+d???A3O?)????Yjܛ?0Ѡ?r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1@1?d?e???q??>s??Ae??Q??Y?]=?1??r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1Eׅ?????_#I??AYl???Z??YG?,?ģ?r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1????yN??J_9????A;6??~??Y??r????r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	?? @???????e??A?JU???Y4?s?륩?r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
w-!?,??/O??RB??A????????Y|+????r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?B ?8???"?:?v???AA.q?????Y?}?<???r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1__?R#4???Z^??6??Aj?:?z??YH?ξ? ??r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??7L???z??{??A$?@?X??YH?ξ? ??r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails11????????1=a???A??)U??Y;m?????r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1J??????IV?F??A?<??- ??Y40??&??r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??Ƅ?K????#?????A??D????Y??%jj??r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1bK??z?????^~????A???????Y?c?~???r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??nIX?????????A-%?I(}??YvP??W??r	train 517*	    ???@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatEIH?m|??!?<{F5[N@)??????1?-??
L@:Preprocessing2T
Iterator::Root::ParallelMapV2Jy??????!???(@)Jy??????1???(@:Preprocessing2E
Iterator::Root?? ?????!???T?4@)膦?????1?|Ys?"!@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????b??!^c???9@)?????b??1^c???9@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?G?`????!{_?j?S@)????????1?u'"@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*@?J?????!!q???@)@?J?????1!q???@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?Đ?L???!gA???%@)?7?櫤?1@?z???
@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap/??w???!?TT̀?+@)B	3m?ʢ?1??K?TJ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?-?b??@I???T?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	u)˹????f?l???4S??!???????	!       "	!       *	!       2$	 ??B??<??????D????!;6??~??:	!       B	!       J$	??k??????r??_3??vP??W??!?}?<???R	!       Z$	??k??????r??_3??vP??W??!?}?<???b	!       JCPU_ONLYY?-?b??@b q???T?X@Y      Y@q?%<WEO??"?
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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