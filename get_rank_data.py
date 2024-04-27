import os
import json
import random
# 获取当前目录下的所有文件和文件夹名称
entries = os.listdir('/ai/ld/remote/mbti/data')

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 过滤出文件，排除文件夹
files = [entry for entry in entries if entry.endswith('.json')]
motion_dict1={'理性的':'decision_thinking','外向的':'energy_extraversion','判断的':'execution_judging','直觉的':'information_intuition'}
motion_dict2={'感性的':'decision_feeling','内向的':'energy_introversion','感知的':'execution_perceiving','感官的':'information_sensing'}
decision_feeling_file=load_json('/ai/ld/remote/mbti/data/zh_decision_feeling.json')
decision_thinking_file=load_json('/ai/ld/remote/mbti/data/zh_decision_thinking.json')
energy_introversion_file=load_json('/ai/ld/remote/mbti/data/zh_energy_introversion.json')
energy_extraversion_file=load_json('/ai/ld/remote/mbti/data/zh_energy_extraversion.json')
execution_judging_file=load_json('/ai/ld/remote/mbti/data/zh_execution_judging.json')
execution_perceiving_file=load_json('/ai/ld/remote/mbti/data/zh_execution_perceiving.json')
information_intuition_file=load_json('/ai/ld/remote/mbti/data/zh_information_intuition.json')
information_sensing_file=load_json('/ai/ld/remote/mbti/data/zh_information_sensing.json')

all_rank_data=[]
for index,type1 in enumerate(motion_dict1.keys()):
    temp_rank_data=[]
    
    good_data=load_json('/ai/ld/remote/mbti/data/zh_'+motion_dict1[type1]+'.json')
    bad_data=load_json('/ai/ld/remote/mbti/data/zh_'+list(motion_dict2.values())[index]+'.json')
    for ind,i in enumerate(good_data):
        random_numbers = [random.choice([0, 1]) for _ in range(3)]
        random_numbers.insert(index,0)
        type_list_temp=[]
        for i_1 in range(4):
            if i_1==index:
                type_list_temp.append(type1)
            else:
                if random_numbers[i_1]==0:
                    type_list_temp.append(list(motion_dict1.keys())[i_1])
                else:
                    type_list_temp.append(list(motion_dict2.keys())[i_1])
        random.shuffle(type_list_temp)
        prompt='请你作为一个{}人，回答以下问题：\n'.format('、'.join(type_list_temp))
        i['output']=[i['output'],bad_data[ind]['output']]
        i['instruction']=prompt+i['instruction']
        i['history']=[]
        temp_rank_data.append(i)
    all_rank_data+=temp_rank_data

for index,type1 in enumerate(motion_dict2.keys()):
    temp_rank_data=[]
    random_numbers = [random.choice([0, 1]) for _ in range(3)]
    prompt='请你作为一个{}人，回答以下问题：'.format(type1)
    good_data=load_json('/ai/ld/remote/mbti/data/zh_'+motion_dict2[type1]+'.json')
    bad_data=load_json('/ai/ld/remote/mbti/data/zh_'+list(motion_dict1.values())[index]+'.json')
    for ind,i in enumerate(good_data):
        random_numbers = [random.choice([0, 1]) for _ in range(3)]
        random_numbers.insert(index,1)
        type_list_temp=[]
        for i_1 in range(4):
            if i_1==index:
                type_list_temp.append(type1)
            else:
                if random_numbers[i_1]==0:
                    type_list_temp.append(list(motion_dict1.keys())[i_1])
                else:
                    type_list_temp.append(list(motion_dict2.keys())[i_1])
        random.shuffle(type_list_temp)
        prompt='请你作为一个{}人，回答以下问题：\n'.format('、'.join(type_list_temp))
        i['output']=[i['output'],bad_data[ind]['output']]
        i['instruction']=prompt+i['instruction']
        i['history']=[]
        temp_rank_data.append(i)
    all_rank_data+=temp_rank_data
    with open('/ai/ld/remote/mbti/data/mbti_rank.json', 'w', encoding='utf-8') as file:
        json.dump(all_rank_data, file, ensure_ascii=False, indent=4)


