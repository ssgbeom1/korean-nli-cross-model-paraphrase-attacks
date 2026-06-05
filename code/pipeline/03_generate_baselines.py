import argparse
import random
import re
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GENERATED_ATTACK_DIR, SAMPLED_SOURCE_DIR, baseline_attack_filename, sampled_source_filename

class BertAttack:

    def __init__(self, model_name='klue/bert-base', top_k=5, max_replace=3):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.top_k = top_k
        self.max_replace = max_replace
        self.mask_token = self.tokenizer.mask_token

    def get_replacement(self, sentence: str, word_idx: int, words: list) -> str:
        import torch
        masked_words = words.copy()
        original_word = masked_words[word_idx]
        masked_words[word_idx] = self.mask_token
        masked_sentence = ' '.join(masked_words)
        inputs = self.tokenizer(masked_sentence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        mask_idx = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_idx) == 0:
            return original_word
        mask_idx = mask_idx[0].item()
        probs = torch.softmax(predictions[0, mask_idx], dim=-1)
        top_k_ids = torch.topk(probs, self.top_k).indices.tolist()
        candidates = []
        for token_id in top_k_ids:
            token = self.tokenizer.decode([token_id]).strip()
            if token and token != original_word and (len(token) > 1) and (not token.startswith('##')):
                candidates.append(token)
        if candidates:
            return random.choice(candidates[:3])
        return original_word

    def attack(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        num_replace = min(self.max_replace, max(1, len(words) // 4))
        replace_indices = random.sample(range(len(words)), min(num_replace, len(words)))
        new_words = words.copy()
        for idx in replace_indices:
            if len(words[idx]) > 1:
                new_word = self.get_replacement(text, idx, words)
                new_words[idx] = new_word
        return ' '.join(new_words)

class BackTranslation:

    def __init__(self):
        try:
            from deep_translator import GoogleTranslator
            self.translator_to_en = GoogleTranslator(source='ko', target='en')
            self.translator_to_ko = GoogleTranslator(source='en', target='ko')
            self.use_deep_translator = True
        except ImportError:
            try:
                from googletrans import Translator
                self.translator = Translator()
                self.use_deep_translator = False
            except ImportError:
                raise ImportError('deep-translator 또는 googletrans를 설치하세요: pip install deep-translator')

    def attack(self, text: str) -> str:
        try:
            if self.use_deep_translator:
                english = self.translator_to_en.translate(text)
                korean = self.translator_to_ko.translate(english)
            else:
                english = self.translator.translate(text, src='ko', dest='en').text
                korean = self.translator.translate(english, src='en', dest='ko').text
            return korean if korean else text
        except Exception as e:
            print(f'번역 에러: {e}')
            return text

class EDA:

    def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1):
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.alpha_rd = alpha_rd
        self.synonyms = {'좋다': ['훌륭하다', '괜찮다', '뛰어나다', '우수하다'], '나쁘다': ['좋지않다', '형편없다', '부족하다'], '크다': ['거대하다', '방대하다', '넓다'], '작다': ['적다', '조그맣다', '소규모다'], '많다': ['다수다', '풍부하다', '넘치다'], '적다': ['부족하다', '드물다', '희소하다'], '빠르다': ['신속하다', '급하다', '재빠르다'], '느리다': ['더디다', '천천히', '완만하다'], '있다': ['존재하다', '있다', '위치하다'], '없다': ['부재하다', '없다', '사라지다'], '하다': ['수행하다', '실시하다', '진행하다'], '되다': ['이루어지다', '완성되다', '달성되다'], '가다': ['이동하다', '향하다', '떠나다'], '오다': ['도착하다', '방문하다', '찾아오다'], '보다': ['관찰하다', '살펴보다', '확인하다'], '말하다': ['이야기하다', '언급하다', '발언하다'], '알다': ['인지하다', '파악하다', '이해하다'], '생각하다': ['사고하다', '판단하다', '고려하다'], '사람': ['인간', '사람들', '인물'], '일': ['업무', '작업', '사안'], '때': ['시기', '시점', '경우'], '것': ['사물', '대상', '물건'], '중요하다': ['핵심적이다', '필수적이다', '긴요하다'], '필요하다': ['요구되다', '필수적이다', '소요되다'], '가능하다': ['할수있다', '실현가능하다', '허용되다'], '어렵다': ['힘들다', '곤란하다', '난해하다'], '쉽다': ['간단하다', '용이하다', '수월하다'], '새롭다': ['신선하다', '참신하다', '새로운'], '오래되다': ['낡다', '구식이다', '오랜'], '높다': ['고위의', '상위의', '높은'], '낮다': ['하위의', '저조하다', '낮은'], '발생하다': ['일어나다', '생기다', '발발하다'], '개발하다': ['발전시키다', '만들다', '구축하다'], '제공하다': ['공급하다', '지원하다', '전달하다'], '사용하다': ['이용하다', '활용하다', '쓰다'], '진행하다': ['수행하다', '실시하다', '추진하다'], '시작하다': ['개시하다', '착수하다', '출발하다'], '마치다': ['완료하다', '끝내다', '종료하다']}
        self.insert_words = ['매우', '정말', '아주', '상당히', '꽤', '다소', '약간', '특히', '주로', '대체로', '일반적으로', '실제로']

    def get_synonym(self, word: str) -> str:
        for key, synonyms in self.synonyms.items():
            if key in word:
                return word.replace(key, random.choice(synonyms))
        return word

    def synonym_replacement(self, words: list, n: int) -> list:
        new_words = words.copy()
        random_indices = list(range(len(words)))
        random.shuffle(random_indices)
        num_replaced = 0
        for idx in random_indices:
            if num_replaced >= n:
                break
            new_word = self.get_synonym(words[idx])
            if new_word != words[idx]:
                new_words[idx] = new_word
                num_replaced += 1
        return new_words

    def random_insertion(self, words: list, n: int) -> list:
        new_words = words.copy()
        for _ in range(n):
            insert_word = random.choice(self.insert_words)
            insert_idx = random.randint(0, len(new_words))
            new_words.insert(insert_idx, insert_word)
        return new_words

    def random_swap(self, words: list, n: int) -> list:
        new_words = words.copy()
        for _ in range(n):
            if len(new_words) < 2:
                break
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = (new_words[idx2], new_words[idx1])
        return new_words

    def random_deletion(self, words: list, p: float) -> list:
        if len(words) == 1:
            return words
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        if len(new_words) == 0:
            return [random.choice(words)]
        return new_words

    def attack(self, text: str) -> str:
        words = text.split()
        num_words = len(words)
        if num_words < 2:
            return text
        n_sr = max(1, int(self.alpha_sr * num_words))
        n_ri = max(1, int(self.alpha_ri * num_words))
        n_rs = max(1, int(self.alpha_rs * num_words))
        augmented_words = words.copy()
        method = random.choice(['sr', 'ri', 'rs', 'rd'])
        if method == 'sr':
            augmented_words = self.synonym_replacement(augmented_words, n_sr)
        elif method == 'ri':
            augmented_words = self.random_insertion(augmented_words, n_ri)
        elif method == 'rs':
            augmented_words = self.random_swap(augmented_words, n_rs)
        elif method == 'rd':
            augmented_words = self.random_deletion(augmented_words, self.alpha_rd)
        return ' '.join(augmented_words)

def main():
    ap = argparse.ArgumentParser(description='베이스라인 패러프레이즈 생성')
    ap.add_argument('--method', required=True, choices=['bert_attack', 'backtranslation', 'eda'], help='베이스라인 방법 선택')
    ap.add_argument('--input', default=str(Path(SAMPLED_SOURCE_DIR) / sampled_source_filename()), help='입력 파일 경로')
    ap.add_argument('--output', default=None, help='출력 파일 경로 (기본: data/02_generated_attacks/<method>_attacks_3000.csv)')
    ap.add_argument('--n', type=int, default=3000, help='처리할 샘플 수')
    ap.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    args = ap.parse_args()
    random.seed(args.seed)
    if args.output is None:
        args.output = str(Path(GENERATED_ATTACK_DIR) / baseline_attack_filename(args.method, args.n))
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f' Loading: {input_path}')
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df = df.head(args.n).copy()
    print(f'   Loaded {len(df)} samples')
    print(f'\n Initializing {args.method}...')
    if args.method == 'bert_attack':
        attacker = BertAttack(model_name='klue/bert-base', top_k=5, max_replace=2)
    elif args.method == 'backtranslation':
        attacker = BackTranslation()
    elif args.method == 'eda':
        attacker = EDA(alpha_sr=0.15, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1)
    else:
        raise ValueError(f'Unknown method: {args.method}')
    print(f'\n Generating paraphrases with {args.method}...')
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        hypothesis = str(row['hypothesis'])
        try:
            attacked = attacker.attack(hypothesis)
            if not attacked or attacked.strip() == '':
                attacked = 'Error'
        except Exception as e:
            print(f'Error at idx {idx}: {e}')
            attacked = 'Error'
        results.append({'id': row.get('id', idx), 'premise': row['premise'], 'hypothesis': hypothesis, 'label': row['label'], 'attacked_hypothesis': attacked, 'generator': f'baseline_{args.method}'})
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print('\n' + '=' * 60)
    print(f'[OK] Saved: {output_path}')
    print(f'   Total: {len(result_df)}')
    error_count = (result_df['attacked_hypothesis'] == 'Error').sum()
    print(f'   Errors: {error_count}')
    print('=' * 60)
    print('\n Samples:')
    for i in range(min(3, len(result_df))):
        row = result_df.iloc[i]
        print(f'\n[{i + 1}]')
        print(f"  원본: {row['hypothesis'][:50]}...")
        print(f"  변형: {row['attacked_hypothesis'][:50]}...")
if __name__ == '__main__':
    main()
