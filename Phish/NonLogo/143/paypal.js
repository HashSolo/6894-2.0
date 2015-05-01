
function coolRedirect(url, msg)
{
   var TARG_ID = "COOL_REDIRECT";
   var DEF_MSG = "";

   if( ! msg )
   {
      msg = DEF_MSG;
   }

   if( ! url )
   {
      throw new Error('You didn\'t include the "url" parameter');
   }


   var e = document.getElementById(TARG_ID);

   if( ! e )
   {
      throw new Error('"COOL_REDIRECT" element id not found');
   }

   var cTicks = parseInt(e.innerHTML);

   var timer = setInterval(function()
   {
      if( cTicks )
      {
         e.innerHTML = --cTicks;
      }
      else
      {
         clearInterval(timer);
         document.body.innerHTML = msg;
         location = url;	  
      }

   }, 1000);
}