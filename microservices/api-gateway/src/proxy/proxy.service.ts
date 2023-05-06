import { Injectable } from '@nestjs/common';
import { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import axios from 'axios';

@Injectable()
export class ProxyService {
  private axiosInstance: AxiosInstance;

  constructor() {
    this.axiosInstance = axios.create();
  }

  async proxyRequest(
    url: string,
    method: string,
    data: any,
  ): Promise<AxiosResponse> {
    const axiosRequestConfig: AxiosRequestConfig = {
      url,
      method,
      data,
    };

    try {
      const response = await this.axiosInstance.request(axiosRequestConfig);
      return response;
    } catch (error) {
      throw new Error('Failed to proxy request');
    }
  }
}
