"use client";

import { useBackendStatus } from '@/hooks/useBackendStatus';
import { Wifi, WifiOff, Loader2 } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

export default function BackendStatus() {
  const { status, lastChecked } = useBackendStatus(30000); // Check every 30 seconds

  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return 'bg-green-500';
      case 'disconnected':
        return 'bg-red-500';
      case 'checking':
        return 'bg-yellow-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'connected':
        return <Wifi className="w-4 h-4 text-green-500" />;
      case 'disconnected':
        return <WifiOff className="w-4 h-4 text-red-500" />;
      case 'checking':
        return <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />;
      default:
        return <Wifi className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return 'Backend connected';
      case 'disconnected':
        return 'Backend disconnected';
      case 'checking':
        return 'Checking connection...';
      default:
        return 'Unknown status';
    }
  };

  const formatLastChecked = () => {
    if (!lastChecked) return 'Never';
    const now = new Date();
    const diff = Math.floor((now.getTime() - lastChecked.getTime()) / 1000);
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return lastChecked.toLocaleTimeString();
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-card/50 border border-border/40 hover:bg-card/80 transition-colors cursor-pointer">
            <div className="relative">
              {getStatusIcon()}
              <div className={`absolute -top-1 -right-1 w-2 h-2 rounded-full ${getStatusColor()} animate-pulse`} />
            </div>
            <span className="text-xs font-medium hidden sm:inline">
              {status === 'connected' ? 'Online' : status === 'disconnected' ? 'Offline' : 'Checking'}
            </span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <div className="text-xs space-y-1">
            <p className="font-semibold">{getStatusText()}</p>
            <p className="text-muted-foreground">Last checked: {formatLastChecked()}</p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
