Eliminating Multi-Audio Feedback Streams In Hangout Session

Abstract

In video conferencing calls managed by computers, often multiple audio streams will occur from multiple computers at the same time. For example, in Google Hangouts multiple people will log into a “Hangout”. The problem is, that in the case where multiple mics are enabled feedback occurs. This results in indiscernible noise that often results in interrupting meetings until members of the hangout have muted their computers. This paper proposes a basic solution to the issue, using Digital Signal Processing, to classify high feedback scenarios. What we propose here requires no training data or machine learning. In an ideal scenario, we would use machine learning for classification, however because of a lack of training data we will resort to algorithms with no prior knowledge of classification. In this paper, we propose a basic algorithm to classify high feedback scenarios. This is not state of the art, as we are not audio signal processing experts, but expect to achieve a relatively successful classification that could conservatively be used for a variety of settings. With improvements, we expect it could be more robust and quicker. 

Architecture

Diagram 1: Stream Flow












Diagram 2: Signal Timeline



Diagram 3: User Flow

The program should mute the incoming hangout (later timestamp) rather than the later hangout.  

Characterization of Audio Feedback Signal


