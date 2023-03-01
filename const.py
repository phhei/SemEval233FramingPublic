import pipeline.aggregator.Average
import pipeline.aggregator.Pooling
import pipeline.encoder.TransformerEncoder
import pipeline.encoder.GnnEncoder
import pipeline.encoder.SimpleNNEncoder
import pipeline.preprocessing.FrameDatasetHuggingface
import pipeline.preprocessing.FrameDatasetWordEmbeddings
import pipeline.preprocessing.FrameDatasetGraph


ARG_STR_TO_MODULE = {
    "aggregator": {
        "None": (None, None),
        "Average": (pipeline.aggregator.Average.Average, pipeline.aggregator.Average.Average.general_argument_parser),
        "HarmonicAverage": (pipeline.aggregator.Average.HarmonicAverage,
                            pipeline.aggregator.Average.HarmonicAverage.general_argument_parser),
        "PoolingMax": (pipeline.aggregator.Pooling.PoolingMax,
                       pipeline.aggregator.Pooling.PoolingMax.general_argument_parser),
        "PoolingMin": (pipeline.aggregator.Pooling.PoolingMin,
                       pipeline.aggregator.Pooling.PoolingMin.general_argument_parser),
        "PoolingSmoothMax": (pipeline.aggregator.Pooling.PoolingSmoothMax,
                             pipeline.aggregator.Pooling.PoolingSmoothMax.smooth_pooling_parser),
        "PoolingSmoothMin": (pipeline.aggregator.Pooling.PoolingSmoothMin,
                             pipeline.aggregator.Pooling.PoolingSmoothMin.smooth_pooling_parser)
    },
    "encoder": {
        "Huggingface": (pipeline.encoder.TransformerEncoder.Huggingface,
                        pipeline.encoder.TransformerEncoder.Huggingface.transformer_parser),
        "GNN": (pipeline.encoder.GnnEncoder.GnnEncoder,
                pipeline.encoder.GnnEncoder.GnnEncoder.GNN_parser),
        "Linear": (pipeline.encoder.SimpleNNEncoder.LinearNNEncoder,
                   pipeline.encoder.SimpleNNEncoder.LinearNNEncoder.linear_parser)
    },
    "preprocessing": {
        "Huggingface": (pipeline.preprocessing.FrameDatasetHuggingface.FrameDatasetForTransformers,
                        pipeline.preprocessing.FrameDatasetHuggingface.FrameDatasetForTransformers.dataset_parser),
        "WordEmbedding": (pipeline.preprocessing.FrameDatasetWordEmbeddings.FrameDatasetForPlainNeuralNets,
                          pipeline.preprocessing.FrameDatasetWordEmbeddings.FrameDatasetForPlainNeuralNets.dataset_parser),
        "GNN": (pipeline.preprocessing.FrameDatasetGraph.FrameDatasetForGnn,
                pipeline.preprocessing.FrameDatasetGraph.FrameDatasetForGnn.dataset_parser)
    }
}
