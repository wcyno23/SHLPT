def sequence_tagging(contents, predictions, targets):
    """BOI type"""
    typedic = {"org": "ORG", "money": "MONEY", "country": "GPE", "time": "TIME", "law": "LAW", "fact": "FAC",
               "thing": "EVENT", "measure": "QUANTITY",
               "order": "ORDINAL", "art": "WORK_OF_ART", "location": "LOC", "language": "LANGUAGE",
               "person": "PERSON",
               "product": "PRODUCT", "num": "CARDINAL", "national": "NORP", "date": "DATE", "per": "PERCENT",
               "mix": "MISC"}
    prediction_results = []
    target_results = []
    for i in range(len(contents)):
        content = contents[i]
        prediction = predictions[i]
        target = targets[i]
        content_low = content.lower()
        content_split = content.split(" ")
        content_low_split = content_low.split(" ")

        prediction_result = ['O' for _ in range(len(content_split))]
        target_result = ['O' for _ in range(len(content_split))]

        if target == 'end' and prediction == 'end':
            prediction_results.append(prediction_result)
            target_results.append(target_result)
            continue

        if len(prediction) > 0 and prediction[-1] == ';':
            prediction = prediction[:-1]

        prediction_split = prediction.split(';')

        if prediction != "end":
            for j in range(len(prediction_split)):
                tokens = prediction_split[j].split("!")
                if len(tokens) != 2:
                    continue
                entity = tokens[0].strip(' ').lower()
                type = tokens[1].strip(' ')
                if type not in typedic:
                    continue
                if content_low.find(entity) == -1:
                    continue
                index = None
                entity_split = entity.split(" ")
                for k in range(len(content_split)):
                    if content_low_split[k] == entity_split[0] or entity_split[0] in content_low_split[k]:
                        is_same = True
                        for l in range(1, len(entity_split)):
                            if content_low_split[k + l] != entity_split[l] and (
                                    entity_split[0] not in content_low_split[k]):
                                is_same = False
                                break
                        if is_same:
                            index = k
                            break
                if index is None:
                    continue
                else:
                    for k in range(index, index + len(entity_split)):
                        if k == index:
                            prediction_result[k] = 'B-' + typedic[type]
                        else:
                            prediction_result[k] = 'I-' + typedic[type]

            if len(target) > 0 and target[-1] == ";":
                target = target[:-1]

            target_split = target.split(";")

            if target != "end":
                for j in range(len(target_split)):
                    tokens = target_split[j].split("!")
                    if len(tokens) != 2:
                        continue
                    entity = tokens[0].strip(" ").lower()
                    type = tokens[1].strip(" ")
                    if type not in typedic:
                        continue
                    if content_low.find(entity) == -1:
                        continue
                    entity_split = entity.split(" ")
                    index = None
                    for k in range(len(content_split)):
                        if content_low_split[k] == entity_split[0] or entity_split[0] in content_low_split[k]:
                            is_same= True
                            for l in range(1, len(entity_split)):
                                if content_low_split[k + l] != entity_split[l] and (
                                        entity_split[0] not in content_low_split[k]):
                                    is_same = False
                                    break
                            if is_same:
                                index = k
                                break
                    if index is None:
                        continue
                    else:
                        for k in range(index, index + len(entity_split)):
                            if k == index:
                                target_result[k] = "B-" + typedic[type]
                            else:
                                target_result[k] = "I-" + typedic[type]
        prediction_results.append(prediction_result)
        target_results.append(target_result)

    return prediction_results, target_results
