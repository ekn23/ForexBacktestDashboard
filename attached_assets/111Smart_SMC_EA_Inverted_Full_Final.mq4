
//+------------------------------------------------------------------+
//|                                             Smart_SMC_EA_Inverted.mq4 |
//|             Inverted Smart Money Concept EA for MT4              |
//+------------------------------------------------------------------+
#property strict

// Input parameters
input double LotSizeM5      = 0.02;
input double LotSizeM30     = 0.05;
input int    Slippage       = 3;
input int    MagicNumber    = 789456;

input int    RSI_Period     = 14;
input double RSI_Overbought = 70.0;
input double RSI_Oversold   = 30.0;

input int    BB_Period      = 20;
input double BB_Deviation   = 2.0;

input double SAR_Step       = 0.02;
input double SAR_Max        = 0.2;

input int    ADX_Period     = 14;
input double ADX_Threshold  = 20.0;

input double SL_Percent     = 2.0;
input double TP_Percent     = 4.0;

bool TradeOpen = false;

//+------------------------------------------------------------------+
int OnInit()
  {
   Print("Inverted SMC EA initialized.");
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
void OnTick()
  {
   if(AccountFreeMargin() < 100) return;

   double lotSize = LotSizeM30;
   if(Period() == PERIOD_M5) lotSize = LotSizeM5;

   double rsi = iRSI(NULL, 0, RSI_Period, PRICE_CLOSE, 0);
   double adx = iADX(NULL, 0, ADX_Period, PRICE_CLOSE, MODE_MAIN, 0);
   double sar = iSAR(NULL, 0, SAR_Step, SAR_Max, 0);
   double bbUpper = iBands(NULL, 0, BB_Period, BB_Deviation, 0, PRICE_CLOSE, MODE_UPPER, 0);
   double bbLower = iBands(NULL, 0, BB_Period, BB_Deviation, 0, PRICE_CLOSE, MODE_LOWER, 0);
   double bbMiddle= iBands(NULL, 0, BB_Period, BB_Deviation, 0, PRICE_CLOSE, MODE_MAIN, 0);

   double close = Close[0];
   double ask = MarketInfo(Symbol(), MODE_ASK);
   double bid = MarketInfo(Symbol(), MODE_BID);

   bool openBuy = false;
   bool openSell = false;

   // Inverted logic
   if(adx < ADX_Threshold && rsi > RSI_Overbought && close > bbUpper && close < sar)
      openBuy = true;

   if(adx < ADX_Threshold && rsi < RSI_Oversold && close < bbLower && close > sar)
      openSell = true;

   // Close existing orders before new one
   for(int i = OrdersTotal() - 1; i >= 0; i--)
     {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         if(OrderMagicNumber() == MagicNumber && OrderSymbol() == Symbol())
           {
            if(OrderType() == OP_BUY && (rsi < 50 || close > sar || close < bbMiddle))
               OrderClose(OrderTicket(), OrderLots(), bid, Slippage, clrYellow);

            if(OrderType() == OP_SELL && (rsi > 50 || close < sar || close > bbMiddle))
               OrderClose(OrderTicket(), OrderLots(), ask, Slippage, clrYellow);
           }
     }

   if(OrdersTotal() == 0)
     {
      double sl = 0;
      double tp = 0;

      if(openBuy)
        {
         sl = close - (SL_Percent / 100.0) * close;
         tp = close + (TP_Percent / 100.0) * close;

         int buyTicket = OrderSend(Symbol(), OP_BUY, lotSize, ask, Slippage, sl, tp,
                                   "Inverted Buy", MagicNumber, 0, clrBlue);
         if(buyTicket > 0) Print("Inverted BUY order placed at ", ask);
        }
      else if(openSell)
        {
         sl = close + (SL_Percent / 100.0) * close;
         tp = close - (TP_Percent / 100.0) * close;

         int sellTicket = OrderSend(Symbol(), OP_SELL, lotSize, bid, Slippage, sl, tp,
                                    "Inverted Sell", MagicNumber, 0, clrRed);
         if(sellTicket > 0) Print("Inverted SELL order placed at ", bid);
        }
     }
  }
//+------------------------------------------------------------------+
