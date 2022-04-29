import torch
from transformers import BertConfig, BertForSequenceClassification
from classification import ClassificationTrainArguments
from transformers import BertTokenizer

if __name__ == '__main__':
    args = ClassificationTrainArguments(
        pretrained_model_name="beomi/kcbert-base",  
        downstream_corpus_name="data", 
        downstream_model_dir="C:/Users/coms/Desktop/04.07/kcbert/result",  
        downstream_corpus_root_dir =  "C:/Users/coms/Desktop/04.07/kcbert/result/Korpora",
        batch_size=32 if torch.cuda.is_available() else 4,
        learning_rate=5e-5,   
        max_seq_length=128,  
        epochs=1,
        tpu_cores=0 if torch.cuda.is_available() else 8, 
        seed=7,
        )

    tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
    )

    fine_tuned_model_ckpt = torch.load(   
        args.downstream_model_checkpoint_fpath,
        map_location=torch.device("cpu")
    )


    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
    )
    
    model = BertForSequenceClassification(pretrained_model_config)


    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

    model.eval()

    def inference_fn(sentence):
        inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
        )
        with torch.no_grad():
            outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})  #inputs를 파이토치 텐서로 바꾼후 모델 계산하기
            prob = outputs.logits.softmax(dim=1)  ## outputs.logit은 soft-max함수에 넣기 전 로짓(logit)형태이다. 
            zero_prob = round(prob[0][0].item(), 4) ## 긍정/부정일 확률을 소수점 4자리로 반올림
            one_prob = round(prob[0][1].item(), 4)
            two_prob = round(prob[0][2].item(), 4)
            three_prob = round(prob[0][3].item(), 4)
            four_prob = round(prob[0][4].item(), 4)
            five_prob = round(prob[0][5].item(), 4)
            six_prob = round(prob[0][6].item(), 4)
        return prob


    
