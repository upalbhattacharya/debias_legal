# README

Embedding generation model using statute statute. Two variations of the 
encoder model are used:
 - BiLSTM
 - BiLSTM-based HAN

 The model is trained as a statute similarity model. The similarity notion used
 is the Jaccard similarity between the set of statutes between cases. 
 Thereafter, the trained model is used to obtain the outputs from the encoder
 module which are used as embeddings of documents/sentences.
