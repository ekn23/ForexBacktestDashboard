//+------------------------------------------------------------------+
//|                                          Simple MA Crossover.mq4 |
//|                        Copyright 2024, Your Strategy Testing Lab |
//|                                             https://yoursite.com |
//+------------------------------------------------------------------+
#property copyright "Your Strategy Testing Lab"
#property link      "https://yoursite.com"
#property version   "1.00"
#property strict

// Input parameters
input int FastMA = 10;        // Fast Moving Average Period
input int SlowMA = 20;        // Slow Moving Average Period
input double LotSize = 0.1;   // Lot Size
input int StopLoss = 100;     // Stop Loss in points
input int TakeProfit = 200;   // Take Profit in points
input int MagicNumber = 12345; // Magic Number

// Global variables
double fastMA, slowMA, prevFastMA, prevSlowMA;
int ticket = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Simple MA Crossover EA initialized");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("Simple MA Crossover EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Calculate moving averages
   fastMA = iMA(Symbol(), PERIOD_CURRENT, FastMA, 0, MODE_SMA, PRICE_CLOSE, 0);
   slowMA = iMA(Symbol(), PERIOD_CURRENT, SlowMA, 0, MODE_SMA, PRICE_CLOSE, 0);
   
   prevFastMA = iMA(Symbol(), PERIOD_CURRENT, FastMA, 0, MODE_SMA, PRICE_CLOSE, 1);
   prevSlowMA = iMA(Symbol(), PERIOD_CURRENT, SlowMA, 0, MODE_SMA, PRICE_CLOSE, 1);
   
   // Check for open positions
   bool hasPosition = false;
   for(int i = 0; i < OrdersTotal(); i++)
   {
      if(OrderSelect(i, SELECT_BY_POS) && OrderMagicNumber() == MagicNumber)
      {
         hasPosition = true;
         break;
      }
   }
   
   // Trading logic
   if(!hasPosition)
   {
      // Golden Cross - Buy signal
      if(prevFastMA <= prevSlowMA && fastMA > slowMA)
      {
         OpenBuy();
      }
      // Death Cross - Sell signal
      else if(prevFastMA >= prevSlowMA && fastMA < slowMA)
      {
         OpenSell();
      }
   }
   else
   {
      // Close opposite positions
      for(int i = OrdersTotal() - 1; i >= 0; i--)
      {
         if(OrderSelect(i, SELECT_BY_POS) && OrderMagicNumber() == MagicNumber)
         {
            if(OrderType() == OP_BUY && prevFastMA >= prevSlowMA && fastMA < slowMA)
            {
               OrderClose(OrderTicket(), OrderLots(), Bid, 3, clrRed);
            }
            else if(OrderType() == OP_SELL && prevFastMA <= prevSlowMA && fastMA > slowMA)
            {
               OrderClose(OrderTicket(), OrderLots(), Ask, 3, clrBlue);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Open Buy Position                                                |
//+------------------------------------------------------------------+
void OpenBuy()
{
   double sl = StopLoss > 0 ? Ask - StopLoss * Point : 0;
   double tp = TakeProfit > 0 ? Ask + TakeProfit * Point : 0;
   
   ticket = OrderSend(Symbol(), OP_BUY, LotSize, Ask, 3, sl, tp, "MA Crossover Buy", MagicNumber, 0, clrGreen);
   
   if(ticket > 0)
      Print("Buy order opened: ", ticket);
   else
      Print("Error opening buy order: ", GetLastError());
}

//+------------------------------------------------------------------+
//| Open Sell Position                                               |
//+------------------------------------------------------------------+
void OpenSell()
{
   double sl = StopLoss > 0 ? Bid + StopLoss * Point : 0;
   double tp = TakeProfit > 0 ? Bid - TakeProfit * Point : 0;
   
   ticket = OrderSend(Symbol(), OP_SELL, LotSize, Bid, 3, sl, tp, "MA Crossover Sell", MagicNumber, 0, clrRed);
   
   if(ticket > 0)
      Print("Sell order opened: ", ticket);
   else
      Print("Error opening sell order: ", GetLastError());
}