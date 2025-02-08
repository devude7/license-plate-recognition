# filter out characters that do not occur on licence plates
def filter_ocr(text):
    new = ''
    for ch in text:
        if ch.isalnum():
            if ch.isupper() or ch.isdigit():
                new += ch
    return new

# compute iou
def bb_intersection_over_union(boxA, boxB):
	
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
     
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
     
	# compute the area of both the prediction and ground-truth
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
     
	# compute the intersection over union 
	iou = interArea / float(boxAArea + boxBArea - interArea)
     
	return iou

# non binary accuracy - compute similarity between strings (result between [0, 1] instead of just 0 or 1)
def diff_accuracy(ocr_lp: str, true_lp: str) -> float:
    
    max_len = max(len(ocr_lp), len(true_lp))
    
    # fill unmatching strings '@'
    ocr_lp = ocr_lp.ljust(max_len, '@')
    true_lp = true_lp.ljust(max_len, '@')
    
    matches = sum(1 for ocr, true in zip(ocr_lp, true_lp) if ocr == true)
    accuracy = matches / max_len
    
    return accuracy