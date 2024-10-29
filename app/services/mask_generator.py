from mobile_sam import MobileSAM  # Mobile-SAM 모델이 제공된다고 가정

# 객체 마스크 생성 함수
def generate_mask(image, selection):
    sam_model = MobileSAM()
    mask = sam_model.generate_mask(image, selection)
    return mask
