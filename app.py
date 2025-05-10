from multiprocessing import freeze_support

# latitude [-90, 90] and longitude [-180, 180)

if __name__ == '__main__':
    import src.main as main

    freeze_support()

# TODO: for model, include has_disaster and season
# TODO: for point generation/data summary, use all disasters within the hexagon, not a point
# TODO: need to remove duplicates in summary_dataset.py
