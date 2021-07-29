# Attentive Multiveiw Neural Network (AMNN) Model


## Introduction

AMNN is framework that can combine different views (representations) of the same input through effective data fusion and attention strategies for ranking purposes. We developed the model to find the most probable diseases that match with clinical descriptions of patients, using data from the [Undiagnosed Diseases Network](https://undiagnosed.hms.harvard.edu/). 

As the following Figure shows, AMNN comprises of three major components:
(a): an attention network that estimates and weights the contribution of each view in the ranking process, 
(b): a fusion network the utilizes intra-view feature interactions to effectively combine query-document representations, and 
(c): a softmax layer at the end that estimates the query-document relevance scores given their combined representations. 

![The architecture of our Attentive Multiview Neural Model (AMNM)](files/amnn.png?raw=true)  
The architecture of our Attentive Multiview Neural Model (AMNM). For simplicity, we illustrate two views only, e.g. (q', d') indicates representations of the texts of a query and a document, and (q", d") indicates representations of the medical codes and concepts associated with the same query and document. f(.) and g(.) indicate attention and fusion functions respectively, and a i indicates the attentive weight of the ith view estimated by the attention sub-network.


## Instructions

Update the read_data() function in utils.py to read data for processing. Detailed instruction are add to the code as comments. 


## Citation 

Hadi Amiri, Mitra Mohtarami, Isaac S. Kohane. [Attentive Multiview Text Representation for Differential Diagnosis](https://aclanthology.org/2021.acl-short.128.pdf). In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL'21).  
[pdf](https://aclanthology.org/2021.acl-short.128.pdf), [supp](https://aclanthology.org/attachments/2021.acl-short.128.OptionalSupplementaryMaterial.pdf), [slides](files/amnn_acl2021.pdf), [talk](files/amnn_565.mp4)
