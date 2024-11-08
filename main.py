import argparse
from model.utils.proxy_call import OpenaiCall


def arg_parse(parser):
    parser.add_argument('--city', type=str, default='shanghai', choices=["hangzhou", "qingdao", "shenzhen","shanghai", "beijing", "changsha", "wuhan"], help='dataset')
    parser.add_argument('--type', type=str, default='main', choices=["zh", "en"], help='')
    return parser.parse_args()


def run(args) -> None:
    if args.type == "zh":
        query = "我想要一个充满历史与艺术气息的行程，逛逛街，看看桥。"
        from model.itinera import ItiNera
    else:
        query = "I would like an itinerary filled with a sense of history and art, with some street wandering and bridge viewing."
        from model.itinera_en import ItiNera
    day_planner = ItiNera(user_reqs=[query], proxy_call=OpenaiCall(), city=args.city, type=args.type)
    itinerary, lookup = day_planner.solve()


if __name__ == '__main__':
    args = arg_parse(argparse.ArgumentParser())
    run(args)