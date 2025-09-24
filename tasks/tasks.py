import random
from typing import Dict, List, Optional

BASE_ROOM_TASKS: Dict[str, List[str]] = {
    "リビング": [
        "リビングのカーテンを開けて",
        "ソファのクッションを整えて",
        "テレビのリモコンを探して",
        "『そこ』のライトを消す",
        "『こっち』の扇風機をつける",
        "テーブルの上の紙を渡して",
        "『近く』のティッシュを渡して",
        "ダイニングの椅子の上の本を運んで",
        "観葉植物に水をあげて",
        "リビングの電気を少し暗くして",
    ],
    "寝室": [
        "ベッドわきのライトを消す",
        "寝室の電気を一番明るくして",
        "寝室の窓を閉める",
        "寝室の椅子にある服を運んで",
        "『ここ』にある靴下を片付けて",
        "寝室の机の上の本を取ってきて",
        "ベッドメイキングをして",
        "枕をふわっと整えて",
    ],
    "廊下": [
        "廊下の明かりを消して",
        "廊下のゴミを拾って",
        "廊下に落ちている靴をそろえて",
        "廊下のランナーをまっすぐに直して",
    ],
    "キッチン": [
        "台所の電気を少し暗くして",
        "キッチンの上のコップを持ってきて",
        "冷蔵庫から『赤いやつ』を取ってきて",
        "シンクの中のフォークを片付けて",
        "コンロの火を止めて",
        "電子レンジを開けて",
        "食洗機の中を確認して",
        "シンクのスポンジをすすいで",
    ],
    "ダイニング": [
        "ダイニングのテーブルを拭いて",
        "テーブルの上の食器を下げて",
        "ダイニングの椅子をきれいに並べて",
        "テーブルにランチョンマットを敷いて",
        "ペンダントライトの明かりを調整して",
    ],
    "玄関": [
        "玄関の鍵を閉めて",
        "玄関の近くにある雑誌を運んで",
        "玄関の靴を整えて",
        "玄関マットのほこりを払って",
        "傘立ての傘をそろえて",
    ],
    "洗面所": [
        "洗面台の鏡を軽く拭いて",
        "洗面台のコップを洗って",
        "タオルを新しいものに交換して",
        "洗面台のライトを点けて",
        "洗濯かごの周りを整えて",
    ],
    "浴室": [
        "バスタブの栓が閉まっているか確認して",
        "浴室の換気扇をつけて",
        "シャンプーボトルをそろえて",
        "浴槽のふたを閉めて",
        "バスマットを干して",
    ],
    "トイレ": [
        "トイレットペーパーの残量を確認して",
        "トイレの換気扇をつけて",
        "トイレマットを整えて",
        "手洗い場のタオルを交換して",
    ],
    "クローゼット": [
        "クローゼットの照明をつけて",
        "クローゼットのハンガーを整えて",
        "クローゼットの床に置かれた箱を片付けて",
        "収納ケースを引き出して中を確認して",
    ],
    "階段": [
        "階段の電気をつけて",
        "階段の手すりを軽く拭いて",
        "階段の途中にある荷物を片付けて",
        "階段下の収納を確認して",
    ],
    "子供部屋": [
        "子供部屋のおもちゃを片付けて",
        "子供部屋の照明をつけて",
        "本棚の絵本をそろえて",
        "学習机の上を整えて",
    ],
    "ガレージ": [
        "ガレージのライトを消して",
        "自転車を所定の位置に戻して",
        "工具が散らかっていないか確認して",
        "シャッターが閉まっているか見てきて",
    ],
    "納屋": [
        "納屋の窓を開けて換気して",
        "納屋の棚に置かれた道具を整えて",
        "納屋のライトをつけて",
        "納屋の床に落ちているものを拾って",
    ],
    "予備室": [
        "部屋のカーテンを開けて",
        "部屋のライトをつけて",
        "机の上のノートをそろえて",
        "部屋の椅子を元の位置に戻して",
    ],
}

ROOM_ALIASES: Dict[str, str] = {
    "リビング": "リビング",
    "LIVING": "リビング",
    "LIVINGROOM": "リビング",
    "LIVING ROOM": "リビング",
    "ROOM": "リビング",
    "寝室": "寝室",
    "BEDROOM": "寝室",
    "キッチン": "キッチン",
    "KITCHEN": "キッチン",
    "ダイニング": "ダイニング",
    "DINING": "ダイニング",
    "DININGROOM": "ダイニング",
    "DINING ROOM": "ダイニング",
    "廊下": "廊下",
    "HALL": "廊下",
    "HALLWAY": "廊下",
    "階段": "階段",
    "STAIRS": "階段",
    "玄関": "玄関",
    "ENTRANCE": "玄関",
    "DOORWAY": "玄関",
    "洗面所": "洗面所",
    "WASHROOM": "洗面所",
    "浴室": "浴室",
    "BATHROOM": "浴室",
    "トイレ": "トイレ",
    "TOILET": "トイレ",
    "クローゼット": "クローゼット",
    "CLOSET": "クローゼット",
    "子供部屋": "子供部屋",
    "KIDSROOM": "子供部屋",
    "ガレージ": "ガレージ",
    "GARAGE": "ガレージ",
    "納屋": "納屋",
    "BARN": "納屋",
    "ROOM1": "予備室",
    "ROOM2": "予備室",
    "ROOM3": "予備室",
    "ROOM4": "予備室",
    "ROOM5": "予備室",
}

DEFAULT_ROOM_TASKS: Dict[str, List[str]] = {
    alias: BASE_ROOM_TASKS[base]
    for alias, base in ROOM_ALIASES.items()
    if base in BASE_ROOM_TASKS
}


def get_tasks_for_room(
    room_name: str,
    tasks_map: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Return the list of tasks associated with the given room name.

    Args:
        room_name: The room identifier selected in the UI. Both Japanese labels
            and room directory names (e.g. ``LIVINGROOM``) are supported.
        tasks_map: Optional mapping that overrides the default task list.

    Returns:
        A list of task strings. Empty when no tasks are registered for the room.
    """

    if not room_name:
        return []

    mapping = tasks_map or DEFAULT_ROOM_TASKS
    normalized = room_name.strip()
    if not normalized:
        return []

    if normalized in mapping:
        return mapping[normalized]

    normalized_upper = normalized.upper()
    if normalized_upper in mapping:
        return mapping[normalized_upper]

    for alias in sorted(ROOM_ALIASES, key=len, reverse=True):
        alias_upper = alias.upper()
        if alias_upper and alias_upper in normalized_upper:
            base = ROOM_ALIASES[alias]
            tasks = (
                mapping.get(alias)
                or mapping.get(alias_upper)
                or mapping.get(base)
                or BASE_ROOM_TASKS.get(base, [])
            )
            if tasks:
                return tasks

    return []


def choose_random_task(
    room_name: str,
    tasks_map: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """Pick a random task for the specified room.

    Args:
        room_name: The room identifier selected in the UI.
        tasks_map: Optional mapping that overrides the default task list.

    Returns:
        A randomly chosen task string, or ``None`` when no tasks are available.
    """

    tasks = get_tasks_for_room(room_name, tasks_map)
    if not tasks:
        return None
    return random.choice(tasks)
