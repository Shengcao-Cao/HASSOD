import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--s-to-ts', action='store_true', help='convert single student model to teacher-student model')
    group.add_argument('--ts-to-s', action='store_true', help='convert teacher-student model to single student model')
    group.add_argument('--ts-to-t', action='store_true', help='convert teacher-student model to single teacher model')

    args = parser.parse_args()

    model = torch.load(args.input, 'cpu')

    if 'optimizer' in model:
        del model['optimizer']
    if 'scheduler' in model:
        del model['scheduler']
    if 'iteration' in model:
        del model['iteration']

    new_state_dict = {}
    if args.s_to_ts:
        for key, value in model['model'].items():
            new_state_dict['modelStudent.' + key] = value
            new_state_dict['modelTeacher.' + key] = value
    elif args.ts_to_s:
        for key, value in model['model'].items():
            if key.startswith('modelStudent.'):
                new_state_dict[key[13:]] = value
    elif args.ts_to_t:
        for key, value in model['model'].items():
            if key.startswith('modelTeacher.'):
                new_state_dict[key[13:]] = value
    else:
        raise ValueError('Conversion direction should be set')

    model['model'] = new_state_dict
    torch.save(model, args.output)
