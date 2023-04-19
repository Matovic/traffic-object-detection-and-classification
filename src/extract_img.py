from pie_data import PIE


def extract_img(pie: PIE) -> None:
    """
    Extract and save only annotated frames.
    :param: pie: PIE object
    """
    pie.extract_and_save_images(extract_frame_type='annotated')
