import torchvision.transforms as transforms
import argparse
import os
from datetime import datetime

import dcsgan.pororo_data as data
from vfid.fid_score import fid_score

def main(args):

    image_transforms = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ref_dataset = data.ImageClfDataset(args.img_ref_dir,
                                    args.imsize,
                                    mode=args.mode,
                                    transform=image_transforms)
    gen_dataset = data.ImageClfDataset(args.img_ref_dir,
                                    args.imsize,
                                    mode=args.mode,
                                    out_img_folder=args.img_gen_dir,
                                    transform=image_transforms)


    fid = fid_score(ref_dataset, gen_dataset, cuda=True, normalize=True, r_cache=os.path.join(args.img_ref_dir, 'fid_cache.npz'), batch_size=1)
    print('Frechet Image Distance: ', fid)
    
    #Moda: log results
    log_file_name = 'fid_results.txt'
    log_file_path = os.path.join(args.img_gen_dir, log_file_name)
    # dd/mm/YY H:M:S
    log_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{args.img_gen_dir}\n{log_time}\nfid score\n\n')
        log_file.write(f'FID score:\t{fid}\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate Frechet Story and Image distance')
    parser.add_argument('--img_ref_dir', type=str, required=True)
    parser.add_argument('--img_gen_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--imsize', type=int, default=64)
    args = parser.parse_args()

    print(args)
    main(args)
