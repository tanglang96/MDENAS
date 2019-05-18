from run_manager import RunConfig


class ImagenetRunConfig(RunConfig):
    def __init__(self, dataset='imagenet', test_batch_size=500, n_worker=32, local_rank=0, world_size=1,
                 darts_gene=None, mobile_gene=None, model_type='gpu', **kwargs):
        super(ImagenetRunConfig, self).__init__(dataset, test_batch_size, local_rank, world_size
                                                )

        self.n_worker = n_worker
        self.darts_gene = darts_gene
        self.mobile_gene = mobile_gene
        self.model_type = model_type

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': 256,
            'test_batch_size': self.test_batch_size,
            'n_worker': self.n_worker,
            'local_rank': self.local_rank,
            'world_size': self.world_size
        }
