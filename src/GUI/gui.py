import PySimpleGUI as sg

# レイアウトの定義
layout = [[sg.Text('テキスト入力してください：')],
          [sg.InputText()],
          [sg.Button('OK'), sg.Button('キャンセル')]]

# ウィンドウの作成
window = sg.Window('テキスト入力', layout)

# イベントループ
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'キャンセル':
        break
    if event == 'OK':
        input_text = values[0]
        sg.popup('入力されたテキスト:', input_text)

# ウィンドウの破棄
window.close()
