import torch
from classification import ClassificationTrainArguments
from utils import set_seed
from utils import set_logger
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from classification import NsmcCorpus, ClassificationDataset, ClassificationTask
from data_utils import data_collator
from trainer import get_trainer







if __name__ == '__main__':
    args = ClassificationTrainArguments(
    pretrained_model_name="beomi/kcbert-base",  
    downstream_corpus_name="data", 
    downstream_model_dir="../../../result",  
    downstream_corpus_root_dir =  "../../../result/Korpora",
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=5e-5,   
    max_seq_length=128,  
    epochs=1,
    tpu_cores=0 if torch.cuda.is_available() else 8, 
    seed=7,
    )

    set_seed(args)
    set_logger(args)



    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )



    corpus = NsmcCorpus()   


    train_dataset = ClassificationDataset(  
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="train",
    )


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False), 
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
    )


    val_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="test",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset), 
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
    )

    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )

    model = BertForSequenceClassification.from_pretrained(
            args.pretrained_model_name,
            config=pretrained_model_config,
    )



    task = ClassificationTask(model, args)
    trainer = get_trainer(args)
    trainer.fit(
    task,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
    )

