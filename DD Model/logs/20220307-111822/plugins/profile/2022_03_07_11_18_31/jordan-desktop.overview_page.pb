?&$	?m@<??oeO[?M????4????!g?v???$	????@2Á6,????b?&??@!V?u?C@"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1M?O/??J???nI??A8?L???YP:?`????r	train 501"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1??4?????X?O0??Ac_??`???Y??_YiR??r	train 502"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1B??^~'??Z?wg???Au ???W??Y???M??r	train 503"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1`?;?????5?؀???A??m?T??Y?P??C???r	train 504"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1l?6???6׆?q??A?p??[???Y?"?????r	train 505"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1'?????????ih??A?OVW???YJ???nI??r	train 506"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1z?]?z+????z??Aq̲'?M??Y*????Χ?r	train 507"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1	M???$???9?m½??A??Ӹ7???Y???w???r	train 508"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1
F{????4???????A?ΡU???Y<FzQ???r	train 509"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails17??­??9?Z?????A|??mTg??Y????ne??r	train 510"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1A?Ρ??y#????Ao?;2V???Yoe??2???r	train 511"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1J?>????-&6׆??Ac??????Y???`???r	train 512"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1$(~??????(?[Z??A?ej?!??Y?ފ?5??r	train 513"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1?D?$??5S"??A}?|?.???Y/??w???r	train 514"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1I?[ϐ??W'g(?x??AyZ~?*O??YE??f?R??r	train 515"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1w?>X??????vhX??A?e???-??YH5???:??r	train 516"r
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails1g?v???B?<?E~??A?*??]???Y?C?ͩd??r	train 517*	fffffȍ@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat|?O?D??!@?Ē?D@)&?v??-??1,	a?U\A@:Preprocessing2T
Iterator::Root::ParallelMapV2?8?@d???!?J*??2@)?8?@d???1?J*??2@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceЛ?T[??!?'?ϋ>'@)Л?T[??1?'?ϋ>'@:Preprocessing2E
Iterator::Root????????!Q?0'?
=@)???????1??@%@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipZ5?????!??3vC?Q@)h?N?????1?0??? @:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*VIddY??!?;??@)VIddY??1?;??@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateH???=??!S?qߗ0@)?I*S?A??1??"'f?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap;m?????!?ӽb(5@)?3?%??1%??#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9'?+?<@I?^?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	v???'U???]>/J??4???????!5?؀???	!       "	!       *	!       2$	X??'?8????ռ??c_??`???!?*??]???:	!       B	!       J$	???,???!????v?<FzQ???!????ne??R	!       Z$	???,???!????v?<FzQ???!????ne??b	!       JCPU_ONLYY'?+?<@b q?^?X@Y      Y@q?᪇?@"?
both?Your program is POTENTIALLY input-bound because 37.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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