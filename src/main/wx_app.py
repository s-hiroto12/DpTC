import wx
import gen_chord_progression

class MyPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.text = wx.TextCtrl(self, -1 , style=wx.TE_PROCESS_ENTER)
        self.text.SetBackgroundColour('gray')
        self.text.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)

        message = wx.StaticText(self, -1, "input initial chord")
        message.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        message.SetForegroundColour(wx.Colour(128, 128, 128))
        
        button = wx.Button(self, -1, 'input')
        button.Bind(wx.EVT_BUTTON, self.OnButtonPress)
        
        box_sizer = wx.BoxSizer(wx.VERTICAL)
        box_sizer.AddSpacer(20)
        box_sizer.Add(message, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        box_sizer.Add(self.text, 0, wx.EXPAND | wx.ALL, 10)
        box_sizer.Add(button, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        self.SetSizer(box_sizer)
    
    def OnButtonPress(self, event):
        input_text = self.text.GetValue()
        print("button input:", input_text)
    
    def OnEnter(self, event):
        input_text = self.text.GetValue()
        print("Enter input:", input_text)

    def show_result(self):
        NotImplemented

class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, id=-1, title='wxPython')
        panel = MyPanel(self)

        

        self.Show()

if __name__ == '__main__':
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()