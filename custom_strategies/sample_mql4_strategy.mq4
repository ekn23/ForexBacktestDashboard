//+------------------------------------------------------------------+
//|                                          Sample RSI Strategy.mq4 |
//|                                   Sample MQL4 Strategy for Demo |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Sample Strategy"
#property link      ""
#property version   "1.00"

// Input parameters
input int RSI_Period = 14;
input double RSI_Oversold = 30.0;
input double RSI_Overbought = 70.0;
input double Lot_Size = 0.05;
input int StopLoss_Pips = 20;
input int TakeProfit_Pips = 30;

// Global variables
int magic_number = 12345;
double current_rsi = 0.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Sample RSI Strategy initialized");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Calculate RSI
   current_rsi = iRSI(Symbol(), Period(), RSI_Period, PRICE_CLOSE, 0);
   
   // Check for buy signal (RSI oversold)
   if(current_rsi < RSI_Oversold && OrdersTotal() == 0)
   {
      double ask = Ask;
      double sl = ask - StopLoss_Pips * Point;
      double tp = ask + TakeProfit_Pips * Point;
      
      OrderSend(Symbol(), OP_BUY, Lot_Size, ask, 3, sl, tp, "RSI Buy", magic_number, 0, clrGreen);
   }
   
   // Check for sell signal (RSI overbought)
   if(current_rsi > RSI_Overbought && OrdersTotal() == 0)
   {
      double bid = Bid;
      double sl = bid + StopLoss_Pips * Point;
      double tp = bid - TakeProfit_Pips * Point;
      
      OrderSend(Symbol(), OP_SELL, Lot_Size, bid, 3, sl, tp, "RSI Sell", magic_number, 0, clrRed);
   }
}

//+------------------------------------------------------------------+