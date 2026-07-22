import argparse

from matcher import create_template, crop_roi, crop_roi_batch, list_templates


def main():
    parser = argparse.ArgumentParser(
        prog='roi-matcher',
        description='Template-matching-based ROI cropping tool')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_create = subparsers.add_parser('create', help='Create a new template with ROI')
    parser_create.add_argument('template_name', help='Name of the template')
    parser_create.add_argument('image_path', help='Path to the template image')

    subparsers.add_parser('list', help='List all saved templates')

    parser_crop = subparsers.add_parser('crop', help='Crop ROI from a single image')
    parser_crop.add_argument('template_name', help='Name of the template')
    parser_crop.add_argument('image_path', help='Path to the input image')
    parser_crop.add_argument('--output', '-o', default=None,
                             help='Output path (default: <input>_cropped.png)')

    parser_batch = subparsers.add_parser('batch', help='Batch crop ROI from a directory')
    parser_batch.add_argument('template_name', help='Name of the template')
    parser_batch.add_argument('input_dir', help='Directory containing input images')
    parser_batch.add_argument('output_dir', help='Directory for output cropped images')

    args = parser.parse_args()

    if args.command == 'create':
        create_template(args.template_name, args.image_path)

    elif args.command == 'list':
        templates = list_templates()
        if templates:
            print('Saved templates:')
            for t in templates:
                print(f'  {t}')
        else:
            print('No templates saved.')

    elif args.command == 'crop':
        cropped = crop_roi(args.image_path, args.template_name)
        if cropped is None:
            print('Error: ROI not matched in image.')
            return
        output_path = args.output
        if output_path is None:
            import os
            base = os.path.splitext(args.image_path)[0]
            output_path = f'{base}_cropped.png'
        cv2 = __import__('cv2')
        cv2.imwrite(output_path, cropped)
        print(f'Cropped ROI saved to {output_path}')

    elif args.command == 'batch':
        crop_roi_batch(args.input_dir, args.template_name, args.output_dir)


if __name__ == '__main__':
    main()
