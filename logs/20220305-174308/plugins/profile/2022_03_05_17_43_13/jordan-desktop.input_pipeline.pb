$	??????@?K?F?S???"M????!?j,a?@$	;zn|?|@^????@?h???@!??j?ן1@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBX}wk??|???G
??1?7? ???A-??\n0??I?]?)?%??Y???V	??rtrain 31"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?t ?????-W?6???1}iƢ???A?+?j???I1^?????Yv?A]?P??rtrain 32"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBl{?%9`??xak????1`x%?s}??A????oa??I[?:?????Y}w+K??rtrain 33"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?j,a?@6??Ң???1?<???A?u6??y@I?q??Q9??Yb??h????rtrain 34"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB8????c@<?$???1?=~o???Aޯ|?y??I׾?^????Y?u??ť??rtrain 35"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB<??k????aMeQX??1Mjh???A?[Z?{??IK?8?????Y????L??rtrain 36"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB@OI_@?O????1v?Kp???A???????IƋ?!r???Yk'JB"m??rtrain 37"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB	????!??9????	??1????q??A?\4d<J??I?6?X??Yd??u???rtrain 38"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB
?Д?~?????J?.	??1W??????A?J???>??ILl>???Yۧ?1???rtrain 39"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBi???2?@E??????1H?C?????A?<dʇ???IS?o*R???Y?? n/??rtrain 40"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBW>???????XR?>???1Eb?????A?EE?N???I???????Y}uU????rtrain 41"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?a???@???????1:?????AO?\???IU????,??Yi8en???rtrain 42"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB???'@@/3l????1???v?
??AD?b*???If???i??Y?a?A
???rtrain 43"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsBB??=P??s?????1???͋??A??FXT???I?RB?????Y?б?J??rtrain 44"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB???????+???}???1?q?????A?wD???II???|@??Y?qo~?D??rtrain 45"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB)>>!;?????!r?z??1?º????A?'??9x??I.?!?????Yr??Q????rtrain 46"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB?"M???????DK??1???(	???A??b??I????????Y??C?l???rtrain 47"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB??e????<g???1???????A}˜.????I((E+????Y^??a?Q??rtrain 48"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsB'??d?V@???H???1W
?\????A|
????I?S??Yh??Y????G??rtrain 49*	?V??@2O
Iterator::Root::Prefetch??lu9%??!?\-|?L@)??lu9%??1?\-|?L@:Preprocessing2E
Iterator::Root?q?????!z?V?hU@)????@-??1?D?_t?<@:Preprocessing2`
)Iterator::Root::Prefetch::MemoryCacheImpl?X5s???!??C?{?@)?X5s???1??C?{?@:Preprocessing2\
%Iterator::Root::Prefetch::MemoryCache??Y?rL??!2?O??,@)NA~6rݤ?1?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?30.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t29.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9n'3c'k@I??JT??T@Q)W???%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?OϬ?m??ӼOD??????DK??!/3l????	!       "$	??C?+????????qy????(	???!?<???*	!       2$	)??????j??Y??-??\n0??!?u6??y@:$	?>V?ې?????HH9?????????!?q??Q9??B	!       J$	8]qh???S??@ ??^??a?Q??!?u??ť??R	!       Z$	8]qh???S??@ ??^??a?Q??!?u??ť??b	!       JGPUYn'3c'k@b q??JT??T@y)W???%@